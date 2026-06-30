"""
SPORTS DATA MODULE
==================
Obtiene datos de fútbol desde football-data.org
API key gratuita: 10 llamadas/minuto, datos históricos completos

Ligas disponibles con plan gratuito:
- PL  = Premier League
- PD  = La Liga (Primera Division)
- BL1 = Bundesliga
- SA  = Serie A
- FL1 = Ligue 1
- CL  = UEFA Champions League
- DED = Eredivisie
- PPL = Primeira Liga
- ELC = Championship
- BSA = Brasileirao
"""

import os
import time
import sqlite3
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("FOOTBALL_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
DB_PATH = "sports_data.db"

HEADERS = {"X-Auth-Token": API_KEY}

# Ligas a recolectar
LEAGUES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}


##########################
### BASE DE DATOS       ###
##########################

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id TEXT PRIMARY KEY,
            competition TEXT,
            season TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            result TEXT,
            status TEXT,
            matchday INTEGER,
            home_team_id TEXT,
            away_team_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            competition TEXT,
            season TEXT,
            matches_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            goals_for INTEGER DEFAULT 0,
            goals_against INTEGER DEFAULT 0,
            goal_diff INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            draw_rate REAL DEFAULT 0,
            avg_goals_for REAL DEFAULT 0,
            avg_goals_against REAL DEFAULT 0,
            home_matches INTEGER DEFAULT 0,
            home_wins INTEGER DEFAULT 0,
            home_win_rate REAL DEFAULT 0,
            away_matches INTEGER DEFAULT 0,
            away_wins INTEGER DEFAULT 0,
            away_win_rate REAL DEFAULT 0,
            form TEXT,
            form_points INTEGER DEFAULT 0,
            clean_sheets INTEGER DEFAULT 0,
            failed_to_score INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team, competition, season)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS upcoming_matches (
            id TEXT PRIMARY KEY,
            competition TEXT,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            matchday INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competition TEXT,
            season TEXT,
            position INTEGER,
            team TEXT,
            team_id TEXT,
            played INTEGER,
            won INTEGER,
            draw INTEGER,
            lost INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_diff INTEGER,
            points INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(competition, season, team)
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Base de datos inicializada")


##########################
### API CALLS          ###
##########################

def api_get(endpoint: str, params: dict = None) -> dict:
    """Llamada a la API con manejo de rate limit"""
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 429:
            print("  ⏳ Rate limit alcanzado, esperando 60s...")
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        r.raise_for_status()
        time.sleep(6)  # máximo 10 llamadas/minuto
        return r.json()
    except Exception as e:
        print(f"  ❌ Error API: {e}")
        return {}


def get_matches(competition: str, season: str = "2024") -> list:
    """Obtiene todos los partidos de una competición y temporada"""
    data = api_get(f"competitions/{competition}/matches",
                   params={"season": season})
    matches = data.get("matches", [])
    played = [m for m in matches if m.get("status") == "FINISHED"]
    print(f"  ✅ {len(played)} partidos jugados en {competition} {season}")
    return played


def get_upcoming(competition: str) -> list:
    """Obtiene próximos partidos"""
    data = api_get(f"competitions/{competition}/matches",
                   params={"status": "SCHEDULED"})
    matches = data.get("matches", [])
    # Solo los próximos 14 días
    cutoff = datetime.now() + timedelta(days=14)
    upcoming = []
    for m in matches:
        try:
            match_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            if match_date.replace(tzinfo=None) <= cutoff:
                upcoming.append(m)
        except Exception:
            pass
    print(f"  ✅ {len(upcoming)} próximos partidos en {competition}")
    return upcoming


def get_standings(competition: str, season: str = "2024") -> list:
    """Obtiene tabla de posiciones"""
    data = api_get(f"competitions/{competition}/standings",
                   params={"season": season})
    standings = data.get("standings", [])
    if standings:
        return standings[0].get("table", [])
    return []


##########################
### GUARDAR DATOS       ###
##########################

def save_match(match: dict, competition: str):
    try:
        score = match.get("score", {})
        full_time = score.get("fullTime", {})
        home_score = full_time.get("home")
        away_score = full_time.get("away")

        if home_score is None or away_score is None:
            return

        if home_score > away_score:
            result = "H"
        elif away_score > home_score:
            result = "A"
        else:
            result = "D"

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO matches
            (id, competition, season, date, home_team, away_team,
             home_score, away_score, result, status, matchday,
             home_team_id, away_team_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(match.get("id")),
            competition,
            str(match.get("season", {}).get("startDate", "")[:4]),
            match.get("utcDate", "")[:10],
            match.get("homeTeam", {}).get("name", ""),
            match.get("awayTeam", {}).get("name", ""),
            home_score,
            away_score,
            result,
            match.get("status", ""),
            match.get("matchday"),
            str(match.get("homeTeam", {}).get("id", "")),
            str(match.get("awayTeam", {}).get("id", ""))
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  ❌ Error guardando partido: {e}")


def save_upcoming(match: dict, competition: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO upcoming_matches
            (id, competition, date, home_team, away_team,
             home_team_id, away_team_id, matchday)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(match.get("id")),
            competition,
            match.get("utcDate", "")[:10],
            match.get("homeTeam", {}).get("name", ""),
            match.get("awayTeam", {}).get("name", ""),
            str(match.get("homeTeam", {}).get("id", "")),
            str(match.get("awayTeam", {}).get("id", "")),
            match.get("matchday")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  ❌ Error guardando próximo partido: {e}")


def save_standings(table: list, competition: str, season: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for row in table:
            c.execute("""
                INSERT OR REPLACE INTO standings
                (competition, season, position, team, team_id,
                 played, won, draw, lost, goals_for, goals_against,
                 goal_diff, points, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                competition, season,
                row.get("position"),
                row.get("team", {}).get("name", ""),
                str(row.get("team", {}).get("id", "")),
                row.get("playedGames"),
                row.get("won"),
                row.get("draw"),
                row.get("lost"),
                row.get("goalsFor"),
                row.get("goalsAgainst"),
                row.get("goalDifference"),
                row.get("points"),
                datetime.now().isoformat()
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  ❌ Error guardando tabla: {e}")


##########################
### CALCULAR STATS      ###
##########################

def calculate_team_stats():
    """Calcula estadísticas detalladas por equipo"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT DISTINCT home_team, competition, season FROM matches")
    teams_home = c.fetchall()
    c.execute("SELECT DISTINCT away_team, competition, season FROM matches")
    teams_away = c.fetchall()
    teams = set([(t[0], t[1], t[2]) for t in teams_home + teams_away])

    for team, competition, season in teams:
        if not team:
            continue

        # Partidos como local
        c.execute("""
            SELECT home_score, away_score, result FROM matches
            WHERE home_team = ? AND competition = ? AND season = ?
            ORDER BY date DESC
        """, (team, competition, season))
        home_matches = c.fetchall()

        # Partidos como visitante
        c.execute("""
            SELECT home_score, away_score, result FROM matches
            WHERE away_team = ? AND competition = ? AND season = ?
            ORDER BY date DESC
        """, (team, competition, season))
        away_matches = c.fetchall()

        total = len(home_matches) + len(away_matches)
        if total == 0:
            continue

        wins = sum(1 for m in home_matches if m[2] == "H") + \
               sum(1 for m in away_matches if m[2] == "A")
        draws = sum(1 for m in home_matches if m[2] == "D") + \
                sum(1 for m in away_matches if m[2] == "D")
        losses = total - wins - draws

        goals_for = sum(m[0] for m in home_matches) + \
                    sum(m[1] for m in away_matches)
        goals_against = sum(m[1] for m in home_matches) + \
                        sum(m[0] for m in away_matches)

        home_wins = sum(1 for m in home_matches if m[2] == "H")
        away_wins = sum(1 for m in away_matches if m[2] == "A")

        clean_sheets = sum(1 for m in home_matches if m[1] == 0) + \
                       sum(1 for m in away_matches if m[0] == 0)
        failed_to_score = sum(1 for m in home_matches if m[0] == 0) + \
                          sum(1 for m in away_matches if m[1] == 0)

        # Forma reciente — últimos 5 partidos
        all_results = []
        for m in home_matches[:5]:
            if m[2] == "H":
                all_results.append("W")
            elif m[2] == "D":
                all_results.append("D")
            else:
                all_results.append("L")
        for m in away_matches[:max(0, 5 - len(all_results))]:
            if m[2] == "A":
                all_results.append("W")
            elif m[2] == "D":
                all_results.append("D")
            else:
                all_results.append("L")

        form = "".join(all_results[:5])
        form_points = sum(3 if r == "W" else 1 if r == "D" else 0
                          for r in all_results[:5])

        c.execute("""
            INSERT OR REPLACE INTO team_stats
            (team, competition, season, matches_played, wins, draws, losses,
             goals_for, goals_against, goal_diff, win_rate, draw_rate,
             avg_goals_for, avg_goals_against,
             home_matches, home_wins, home_win_rate,
             away_matches, away_wins, away_win_rate,
             form, form_points, clean_sheets, failed_to_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            team, competition, season, total, wins, draws, losses,
            goals_for, goals_against, goals_for - goals_against,
            wins / total, draws / total,
            goals_for / total, goals_against / total,
            len(home_matches), home_wins,
            home_wins / len(home_matches) if home_matches else 0,
            len(away_matches), away_wins,
            away_wins / len(away_matches) if away_matches else 0,
            form, form_points, clean_sheets, failed_to_score,
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()
    print("✅ Estadísticas calculadas")


##########################
### REPORTE            ###
##########################

def print_upcoming_with_stats():
    """Muestra próximos partidos con estadísticas"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT date, home_team, away_team, competition
        FROM upcoming_matches
        ORDER BY date ASC LIMIT 20
    """)
    upcoming = c.fetchall()

    print(f"\n{'═' * 70}")
    print(f"  PRÓXIMOS PARTIDOS")
    print(f"{'═' * 70}")

    for date, home, away, competition in upcoming:
        print(f"\n  {competition} | {date}")
        print(f"  {home} vs {away}")

        for team, role in [(home, "🏠"), (away, "✈️ ")]:
            c.execute("""
                SELECT win_rate, avg_goals_for, avg_goals_against,
                       home_win_rate, away_win_rate, form, form_points,
                       matches_played, clean_sheets
                FROM team_stats WHERE team = ?
                ORDER BY updated_at DESC LIMIT 1
            """, (team,))
            stats = c.fetchone()
            if stats:
                wr, gf, ga, hwr, awr, form, fp, mp, cs = stats
                print(f"  {role} {team}: {mp}PJ | "
                      f"Win%:{wr*100:.0f}% | "
                      f"GF:{gf:.1f} GA:{ga:.1f} | "
                      f"Forma:{form}({fp}pts) | "
                      f"CS:{cs}")

    conn.close()


##########################
### MAIN               ###
##########################

def collect_all_data():
    """Recolecta todos los datos"""
    if not API_KEY:
        print("❌ Falta FOOTBALL_API_KEY en el .env")
        return

    init_db()
    seasons = ["2024", "2023"]

    for code, name in LEAGUES.items():
        print(f"\n📊 Recolectando {name} ({code})...")

        for season in seasons:
            print(f"  Temporada {season}...")
            matches = get_matches(code, season)
            for m in matches:
                save_match(m, code)

        print(f"  Próximos partidos...")
        upcoming = get_upcoming(code)
        for m in upcoming:
            save_upcoming(m, code)

        print(f"  Tabla de posiciones...")
        table = get_standings(code)
        save_standings(table, code, "2024")

    print("\n⚙️ Calculando estadísticas...")
    calculate_team_stats()
    print_upcoming_with_stats()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM matches")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM upcoming_matches")
    upcoming_count = c.fetchone()[0]
    conn.close()

    print(f"\n✅ Listo: {total:,} partidos históricos | {upcoming_count} próximos")


if __name__ == "__main__":
    collect_all_data()
