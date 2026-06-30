import os
import time
import pickle
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv
from py_clob_client_v2 import ClobClient, MarketOrderArgs, OrderType, Side
from sports_model import predict_match

load_dotenv()

PRIVATE_KEY = os.environ.get("PRIVATE_KEY")
FUNDER_ADDRESS = os.environ.get("FUNDER_ADDRESS")
BET_AMOUNT = float(os.environ.get("BET_AMOUNT", 2.0))
DRY_RUN = os.environ.get("DRY_RUN", "True").lower() == "true"

# Configuración de estrategia
MIN_MODEL_CONFIDENCE = 0.55   # mínimo 55% de confianza del modelo
MIN_EDGE = 0.05               # mínimo 5% de ventaja vs precio
MIN_WHALE_BET = 500           # considerar ballena si apuesta > $500
WHALE_BONUS = 0.03            # bajar umbral de edge si hay ballena

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
DB_PATH = "sports_data.db"

already_bet = set()

### CLIENT  ###

def get_clob_client():
    client = ClobClient(
        CLOB_API,
        key=PRIVATE_KEY,
        chain_id=137,
        signature_type=2,
        funder=FUNDER_ADDRESS
    )
    creds = client.derive_api_key()
    client.set_api_creds(creds)
    return client


def get_token_price(token_id: str, side: str = "buy") -> float:
    try:
        r = requests.get(
            f"{CLOB_API}/price",
            params={"token_id": token_id, "side": side},
            timeout=5
        )
        r.raise_for_status()
        return float(r.json().get("price", 0))
    except Exception:
        return 0.0

### WHALE TRACKER ###

def get_recent_large_trades(min_amount: float = 500) -> list:
    """Detecta apuestas grandes recientes"""
    try:
        r = requests.get(
            f"{DATA_API}/activity",
            params={"limit": 50},
            timeout=10
        )
        r.raise_for_status()
        trades = r.json()

        whales = []
        for trade in trades:
            if trade.get("type") != "TRADE":
                continue
            size = float(trade.get("usdcSize", 0) or 0)
            if size >= min_amount:
                whales.append({
                    "title": trade.get("title", ""),
                    "outcome": trade.get("outcome", ""),
                    "side": trade.get("side", ""),
                    "size": size,
                    "price": float(trade.get("price", 0) or 0),
                    "asset": trade.get("asset", ""),
                    "timestamp": trade.get("timestamp", ""),
                })
        return whales
    except Exception as e:
        print(f"   Error whale tracker: {e}")
        return []


def find_whale_signal(market_title: str, home_team: str,
                       away_team: str) -> dict | None:
    """Busca si hay ballenas apostando en un partido específico"""
    whales = get_recent_large_trades(MIN_WHALE_BET)

    for whale in whales:
        title = whale["title"].lower()
        if home_team.lower()[:6] in title or away_team.lower()[:6] in title:
            print(f"   Ballena detectada: ${whale['size']:.0f} en {whale['outcome']}")
            return whale
    return None


### BUSCAR MERCADOS ###

def find_polymarket_match(home_team: str, away_team: str) -> dict | None:
    """Busca el mercado para un partido"""
    try:
        # Buscar por nombre del equipo
        for team in [home_team, away_team]:
            words = team.split()[:2]
            query = " ".join(words)

            r = requests.get(
                f"{GAMMA_API}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 20,
                },
                timeout=10
            )
            r.raise_for_status()
            markets = r.json()

            for market in markets:
                question = market.get("question", "").lower()
                if (home_team.lower()[:5] in question or
                        away_team.lower()[:5] in question):
                    # Verificar que es un mercado de ganador
                    if any(kw in question for kw in ["win", "beat", "vs", "winner"]):
                        return market

        return None
    except Exception as e:
        print(f"   Error buscando mercado: {e}")
        return None


def get_market_tokens(market: dict, predicted_result: str,
                       home_team: str, away_team: str) -> tuple:
    """Obtiene el token correcto según la predicción"""
    outcomes = market.get("outcomes", "[]")
    clob_token_ids = market.get("clobTokenIds", [])

    if isinstance(outcomes, str):
        import json
        outcomes = json.loads(outcomes)

    token_id = None
    outcome_name = None

    for i, outcome in enumerate(outcomes):
        outcome_lower = outcome.lower()
        if predicted_result == "H" and home_team.lower()[:5] in outcome_lower:
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else None
            outcome_name = outcome
            break
        elif predicted_result == "A" and away_team.lower()[:5] in outcome_lower:
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else None
            outcome_name = outcome
            break

    return token_id, outcome_name



### CALCULAR EDGE  ###

def calculate_edge(model_prob: float, token_price: float) -> float:
    """
    Calcula la ventaja esperada.
    Edge = probabilidad modelo - precio token
    Si edge > 0, tenemos ventaja.
    """
    return model_prob - token_price


### EJECUTAR APUESTA ###

def place_bet(token_id: str, amount: float):
    client = get_clob_client()
    order = MarketOrderArgs(
        token_id=token_id,
        amount=amount,
        side=Side.BUY,
        order_type=OrderType.FOK
    )
    client.create_and_post_market_order(order)

### ANALIZAR PARTIDO  ###

def analyze_match(date: str, home_team: str,
                   away_team: str, competition: str):
    """Analiza un partido y decide si apostar"""

    match_key = f"{date}_{home_team}_{away_team}"
    if match_key in already_bet:
        return

    print(f"\n{'─' * 60}")
    print(f"   {home_team} vs {away_team} | {competition} | {date}")
    print(f"{'─' * 60}")

    # 1. Predicción del modelo
    prediction = predict_match(home_team, away_team, competition)
    if not prediction:
        print("   Sin predicción disponible")
        return

    prob_home = prediction["prob_home_win"]
    prob_away = prediction["prob_away_win"]
    prob_draw = prediction["prob_draw"]
    predicted = prediction["predicted_result"]
    confidence = prediction["confidence_result"]

    if confidence < MIN_MODEL_CONFIDENCE:
        print(f"   Confianza baja ({confidence*100:.1f}%). Saltando...")
        return

    # 2. Buscar mercado en Polymarket
    market = find_polymarket_match(home_team, away_team)
    if not market:
        print(f"   Mercado no encontrado en Polymarket")
        return

    print(f"   Mercado encontrado: {market.get('question', '')[:50]}")

    # 3. Obtener token correcto
    token_id, outcome_name = get_market_tokens(
        market, predicted, home_team, away_team
    )

    if not token_id:
        print(f"   Token no encontrado para resultado {predicted}")
        return

    # 4. Precio del token
    token_price = get_token_price(token_id, "buy")
    if token_price <= 0 or token_price >= 0.95:
        print(f"   Precio del token inválido: {token_price*100:.1f}¢")
        return

    # 5. Calcular edge
    model_prob = prob_home if predicted == "H" else prob_away
    edge = calculate_edge(model_prob, token_price)

    print(f"\n   Análisis:")
    print(f"     Predicción modelo: {predicted} ({model_prob*100:.1f}%)")
    print(f"     Precio token:      {token_price*100:.1f}¢")
    print(f"     Edge:              {edge*100:.1f}%")

    # 6. Whale tracker
    whale = find_whale_signal(
        market.get("question", ""), home_team, away_team
    )
    effective_min_edge = MIN_EDGE
    if whale:
        # Si la ballena apuesta en la misma dirección, bajamos el umbral
        whale_outcome = whale.get("outcome", "").lower()
        if home_team.lower()[:5] in whale_outcome and predicted == "H":
            effective_min_edge = MIN_EDGE - WHALE_BONUS
            print(f"   Ballena confirma predicción. Edge mínimo reducido a {effective_min_edge*100:.1f}%")
        elif away_team.lower()[:5] in whale_outcome and predicted == "A":
            effective_min_edge = MIN_EDGE - WHALE_BONUS
            print(f"   Ballena confirma predicción. Edge mínimo reducido a {effective_min_edge*100:.1f}%")

    # 7. Decisión
    if edge < effective_min_edge:
        print(f"\n  ❌ Edge insuficiente ({edge*100:.1f}% < {effective_min_edge*100:.1f}%). No apostar.")
        return

    print(f"\n   APOSTAR: ${BET_AMOUNT:.2f} en {outcome_name} @ {token_price*100:.1f}¢")
    print(f"     Edge: {edge*100:.1f}% | Confianza: {confidence*100:.1f}%")

    if whale:
        print(f"      Confirmado por ballena (${whale['size']:.0f})")

    if DRY_RUN:
        print(f"   DRY RUN — no se ejecuta")
    else:
        try:
            place_bet(token_id, BET_AMOUNT)
            print(f"   Apuesta ejecutada")
            already_bet.add(match_key)
        except Exception as e:
            print(f"   Error ejecutando apuesta: {e}")


###   MAIN   ###

def main():
    print("\n" + "═" * 60)
    print("  SPORTS BETTING BOT — POLYMARKET")
    print("═" * 60)
    print(f"  Confianza mínima modelo: {MIN_MODEL_CONFIDENCE*100:.0f}%")
    print(f"  Edge mínimo: {MIN_EDGE*100:.0f}%")
    print(f"  Whale mínimo: ${MIN_WHALE_BET}")
    print(f"  Monto por apuesta: ${BET_AMOUNT}")
    print(f"  Modo: {'DRY RUN ' if DRY_RUN else 'LIVE '}")
    print("═" * 60 + "\n")

    while True:
        try:
            print(f"\n Revisando próximos partidos... {datetime.now().strftime('%H:%M:%S')}")

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT date, home_team, away_team, competition
                FROM upcoming_matches
                WHERE date >= date('now')
                ORDER BY date ASC
                LIMIT 20
            """)
            upcoming = c.fetchall()
            conn.close()

            for date, home, away, competition in upcoming:
                analyze_match(date, home, away, competition)
                time.sleep(2)

            print(f"\n Próxima revisión en 30 minutos...")
            time.sleep(1800)

        except KeyboardInterrupt:
            print("\n\n Bot detenido.")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
