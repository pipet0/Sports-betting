"""
SPORTS MODEL
============
Modelo predictivo de resultados de fútbol usando LightGBM.
Predice:
1. Resultado del partido (Local gana / Empate / Visitante gana)
2. Over/Under 2.5 goles
"""

import sqlite3
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

DB_PATH = "sports_data.db"
MODEL_PATH = "sports_model.pkl"

### FEATURE ENGINEERING ##

def get_team_features(team: str, competition: str,
                       is_home: bool, conn) -> dict:
    """Extrae features de un equipo"""
    c = conn.cursor()

    c.execute("""
        SELECT win_rate, draw_rate, avg_goals_for, avg_goals_against,
               home_win_rate, away_win_rate, form_points,
               matches_played, clean_sheets, failed_to_score,
               goal_diff, goals_for, goals_against
        FROM team_stats
        WHERE team = ? AND competition = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (team, competition))
    stats = c.fetchone()

    if not stats:
        return {}

    wr, dr, gf, ga, hwr, awr, fp, mp, cs, fts, gd, tgf, tga = stats

    prefix = "home_" if is_home else "away_"
    features = {
        f"{prefix}win_rate": wr or 0,
        f"{prefix}draw_rate": dr or 0,
        f"{prefix}avg_goals_for": gf or 0,
        f"{prefix}avg_goals_against": ga or 0,
        f"{prefix}home_win_rate": hwr or 0,
        f"{prefix}away_win_rate": awr or 0,
        f"{prefix}form_points": fp or 0,
        f"{prefix}matches_played": mp or 0,
        f"{prefix}clean_sheets_pct": (cs / mp) if mp else 0,
        f"{prefix}failed_to_score_pct": (fts / mp) if mp else 0,
        f"{prefix}goal_diff_avg": (gd / mp) if mp else 0,
    }

    # Tabla de posiciones
    c.execute("""
        SELECT position, points, played
        FROM standings
        WHERE team = ? AND competition = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (team, competition))
    standing = c.fetchone()

    if standing:
        pos, pts, played = standing
        features[f"{prefix}position"] = pos or 20
        features[f"{prefix}points_per_game"] = (pts / played) if played else 0
    else:
        features[f"{prefix}position"] = 20
        features[f"{prefix}points_per_game"] = 0

    return features


def build_dataset() -> pd.DataFrame:
    """Construye el dataset de entrenamiento"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT id, competition, home_team, away_team,
               home_score, away_score, result
        FROM matches
        WHERE result IS NOT NULL
        AND home_team != '' AND away_team != ''
        ORDER BY date ASC
    """)
    matches = c.fetchall()

    rows = []
    for match_id, competition, home, away, hs, as_, result in matches:
        home_features = get_team_features(home, competition, True, conn)
        away_features = get_team_features(away, competition, False, conn)

        if not home_features or not away_features:
            continue

        row = {**home_features, **away_features}

        # Features comparativas
        row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]
        row["goals_for_diff"] = row["home_avg_goals_for"] - row["away_avg_goals_for"]
        row["goals_against_diff"] = row["home_avg_goals_against"] - row["away_avg_goals_against"]
        row["form_diff"] = row["home_form_points"] - row["away_form_points"]
        row["position_diff"] = row["away_position"] - row["home_position"]
        row["total_avg_goals"] = row["home_avg_goals_for"] + row["away_avg_goals_for"]

        # Targets
        row["result"] = result  # H, D, A
        row["over_25"] = 1 if (hs + as_) > 2.5 else 0
        row["competition"] = competition

        rows.append(row)

    conn.close()

    df = pd.DataFrame(rows)
    print(f" Dataset: {len(df)} partidos con features")
    print(f"   Distribución resultados: {df['result'].value_counts().to_dict()}")
    print(f"   Over 2.5: {df['over_25'].mean()*100:.1f}%")
    return df

### ENTRENAMIENTO       ###

def train_models(df: pd.DataFrame):
    """Entrena modelo de resultado y over/under"""
    feature_cols = [c for c in df.columns
                    if c not in ["result", "over_25", "competition"]]

    X = df[feature_cols].fillna(0)

    # --- Modelo 1: Resultado (H/D/A) ---
    le = LabelEncoder()
    y_result = le.fit_transform(df["result"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_result, test_size=0.2, random_state=42, stratify=y_result
    )

    model_result = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model_result.fit(X_train, y_train)
    y_pred = model_result.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Modelo Resultado — Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred,
                                  target_names=le.classes_))

    # --- Modelo 2: Over/Under 2.5 ---
    y_over = df["over_25"]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_over, test_size=0.2, random_state=42
    )

    model_over = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model_over.fit(X_train2, y_train2)
    y_pred2 = model_over.predict(X_test2)
    acc2 = accuracy_score(y_test2, y_pred2)
    print(f"\n Modelo Over/Under — Accuracy: {acc2*100:.1f}%")
    print(classification_report(y_test2, y_pred2,
                                  target_names=["Under 2.5", "Over 2.5"]))

    # Feature importance
    importance = pd.Series(
        model_result.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    print(f"\n Top 10 features más importantes:")
    print(importance.head(10).to_string())

    # Guardar modelos
    models = {
        "result": model_result,
        "over": model_over,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "accuracy_result": acc,
        "accuracy_over": acc2,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"\n Modelos guardados en {MODEL_PATH}")
    return models

### PREDICCIÓN         ###

def predict_match(home_team: str, away_team: str,
                   competition: str) -> dict:
    """Predice el resultado de un partido"""
    try:
        with open(MODEL_PATH, "rb") as f:
            models = pickle.load(f)
    except FileNotFoundError:
        print(" Modelo no encontrado. Entrena primero con train()")
        return {}

    conn = sqlite3.connect(DB_PATH)
    home_features = get_team_features(home_team, competition, True, conn)
    away_features = get_team_features(away_team, competition, False, conn)
    conn.close()

    if not home_features or not away_features:
        print(f" No hay datos para {home_team} o {away_team}")
        return {}

    row = {**home_features, **away_features}
    row["win_rate_diff"] = row["home_win_rate"] - row["away_win_rate"]
    row["goals_for_diff"] = row["home_avg_goals_for"] - row["away_avg_goals_for"]
    row["goals_against_diff"] = row["home_avg_goals_against"] - row["away_avg_goals_against"]
    row["form_diff"] = row["home_form_points"] - row["away_form_points"]
    row["position_diff"] = row["away_position"] - row["home_position"]
    row["total_avg_goals"] = row["home_avg_goals_for"] + row["away_avg_goals_for"]

    feature_cols = models["feature_cols"]
    X = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    # Predicción resultado
    le = models["label_encoder"]
    result_probs = models["result"].predict_proba(X)[0]
    result_classes = le.classes_
    result_pred = le.inverse_transform([np.argmax(result_probs)])[0]

    prob_dict = dict(zip(result_classes, result_probs))

    # Predicción over/under
    over_prob = models["over"].predict_proba(X)[0][1]

    prediction = {
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "prob_home_win": prob_dict.get("H", 0),
        "prob_draw": prob_dict.get("D", 0),
        "prob_away_win": prob_dict.get("A", 0),
        "predicted_result": result_pred,
        "prob_over_25": over_prob,
        "prob_under_25": 1 - over_prob,
        "predicted_goals": "Over 2.5" if over_prob > 0.5 else "Under 2.5",
        "confidence_result": max(result_probs),
        "confidence_goals": max(over_prob, 1 - over_prob),
    }

    print(f"\n{'═' * 60}")
    print(f"  PREDICCIÓN: {home_team} vs {away_team}")
    print(f"  Competición: {competition}")
    print(f"{'═' * 60}")
    print(f"   {home_team} gana:  {prediction['prob_home_win']*100:.1f}%")
    print(f"   Empate:           {prediction['prob_draw']*100:.1f}%")
    print(f"    {away_team} gana: {prediction['prob_away_win']*100:.1f}%")
    print(f"   Over 2.5 goles:  {prediction['prob_over_25']*100:.1f}%")
    print(f"\n   Predicción: {result_pred} | {prediction['predicted_goals']}")
    print(f"   Confianza resultado: {prediction['confidence_result']*100:.1f}%")
    print(f"{'═' * 60}")

    return prediction


def predict_all_upcoming():
    """Predice todos los próximos partidos"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT date, home_team, away_team, competition
        FROM upcoming_matches
        ORDER BY date ASC
    """)
    upcoming = c.fetchall()
    conn.close()

    predictions = []
    for date, home, away, competition in upcoming:
        print(f"\n  Prediciendo {home} vs {away}...")
        pred = predict_match(home, away, competition)
        if pred:
            pred["date"] = date
            predictions.append(pred)

    return predictions

### MAIN    ###

def train():
    """Entrena los modelos"""
    print(" Construyendo dataset...")
    df = build_dataset()

    if len(df) < 50:
        print("❌ Pocos datos para entrenar. Ejecuta sports_data.py primero.")
        return

    print(" Entrenando modelos...")
    models = train_models(df)
    return models


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_all_upcoming()
    else:
        train()
