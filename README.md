# Weather Advisory Support Bot

A conversational weather-safety assistant that answers questions like:

- *"Is it safe to cycle today?"*
- *"Should I take my child to the park?"*
- *"Is today a good day for a picnic?"*
- *"Can I ride my scooter this afternoon?"*
- *"Should I go for a run?"*

The system uses **live weather data** and **never invents safety advice**.

> **Core principle:** The LLM interprets language but does not determine safety policy. Weather numbers originate only from validated WeatherFacts. Policy decisions are deterministic.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Architecture](#2-architecture)
3. [LangGraph Graph](#3-langgraph-graph)
4. [Node Responsibilities](#4-node-responsibilities)
5. [LLM Boundary](#5-llm-boundary)
6. [Weather API](#6-weather-api)
7. [WeatherFacts](#7-weatherfacts)
8. [SOP Schema](#8-sop-schema)
9. [SOP Library](#9-sop-library)
10. [Policy Matching](#10-policy-matching)
11. [Conflict Resolution](#11-conflict-resolution)
12. [Session Memory](#12-session-memory)
13. [Failure Handling](#13-failure-handling)
14. [Add-a-Policy Workflow](#14-add-a-policy-workflow)
15. [Setup](#15-setup)
16. [Running the Frontend](#16-running-the-frontend)
17. [Running Tests](#17-running-tests)
18. [Running Evaluations](#18-running-evaluations)
19. [Limitations](#19-limitations)

---

## 1. Problem

Weather-advisory chatbots risk giving unsafe advice if the LLM is allowed to decide what is safe. This project addresses that by making every safety decision deterministic, traceable to a specific SOP (Safety Operating Procedure), and grounded in validated live weather data.

---

## 2. Architecture

```
User Message
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                     LangGraph Graph                      │
│                                                         │
│  parse_intent ──► resolve_location ──► fetch_weather    │
│      │                  │                   │           │
│    (LLM)          (Open-Meteo         (Open-Meteo       │
│  language          Geocoding)          Forecast)        │
│  understanding         │                   │           │
│      │            ── failure ──►    ── failure ──►     │
│      │            error_response    error_response     │
│      ▼                                    │            │
│  extract_facts ◄──────────────────────────┘            │
│      │                                                  │
│      ▼                                                  │
│  match_sops (deterministic, no LLM)                    │
│      │                                                  │
│      ├── no match ──► no_match_response                │
│      ▼                                                  │
│  resolve_policy (deterministic, no LLM)                │
│      │                                                  │
│      ▼                                                  │
│  compose_answer (template rendering, no LLM)           │
│      │                                                  │
│      ▼                                                  │
│   final_answer                                         │
└─────────────────────────────────────────────────────────┘
```

**Component roles:**

| Component | Role |
|---|---|
| LLM (OpenAI) | Natural language understanding, closed-vocab classification only |
| Open-Meteo Geocoding | City/place → (lat, lon) |
| Open-Meteo Forecast | Live weather data |
| WeatherFacts | Typed container for all weather numbers |
| SOP YAML | Externalized safety policy library |
| Policy evaluator | Deterministic AST condition evaluation |
| Policy matcher | Iterates all SOPs, checks applicability + conditions |
| Policy selector | Conflict resolution: override > severity > priority |
| LangGraph | Orchestration with real conditional branching |
| Streamlit | Minimal chat frontend |

---

## 3. LangGraph Graph

The graph is defined in [`app/graph/graph.py`](app/graph/graph.py).

### Nodes

```
START → parse_intent → resolve_location → fetch_weather → extract_facts
        → match_sops → resolve_policy → compose_answer → END

Failure branches:
  resolve_location failure → error_response → END
  fetch_weather failure    → error_response → END
  extract_facts failure    → error_response → END
  match_sops (no match)    → no_match_response → END
```

### Conditional Edges

All branches are **real LangGraph conditional edges**:

- `route_after_location`: location failure → `error_response`
- `route_after_weather`: weather failure → `error_response`
- `route_after_extract`: extract failure → `error_response`
- `route_after_match`: no matches → `no_match_response`

---

## 4. Node Responsibilities

| Node | File | Description |
|---|---|---|
| `parse_intent` | `nodes/parse_intent.py` | LLM call; returns closed-vocab intent |
| `resolve_location` | `nodes/resolve_location.py` | Geocodes location text to (lat, lon) |
| `fetch_weather` | `nodes/fetch_weather.py` | Calls Open-Meteo forecast API |
| `extract_facts` | `nodes/extract_facts.py` | Validates WeatherFacts presence |
| `match_sops` | `nodes/match_sops.py` | Deterministic SOP matching |
| `resolve_policy` | `nodes/resolve_policy.py` | Conflict resolution |
| `compose_answer` | `nodes/compose_answer.py` | Template rendering |
| `error_response` | `nodes/error_response.py` | Failure message |
| `no_match_response` | `nodes/no_match_response.py` | No-policy message |

---

## 5. LLM Boundary

**The LLM interprets language but does not determine safety policy.**

### What the LLM MAY do

- Understand natural language
- Classify user requests into a **closed vocabulary**
- Extract location, activity category, mode, group, and time context
- Recognize paraphrased activity descriptions (e.g., "jog" → `running`)
- Detect follow-up queries (e.g., "what about this evening?")

### What the LLM MUST NOT do

- Return SOP IDs
- Invent safety thresholds
- Select severity levels
- Make safety recommendations
- Determine which SOP wins
- Invent or guess weather data

### Closed vocabulary

The LLM output is validated by Pydantic with strict allowed values:

- **activity_categories**: `outdoor_exercise`, `outdoor_recreation`, `travel`, `vulnerable_groups`, `water_activities`, `general`
- **mode**: `running`, `cycling`, `walking`, `motorbike`, `scooter`, `car`, `bus`, `train`, `swimming`, `boating`, `null`
- **group**: `children`, `elderly`, `general_public`, `null`
- **time_context**: `current`, `morning`, `afternoon`, `evening`, `night`, `today`, `tomorrow`

Any value outside these sets is rejected or sanitized.

---

## 6. Weather API

Uses **Open-Meteo** (free, no API key required):

| Purpose | URL |
|---|---|
| Geocoding | `https://geocoding-api.open-meteo.com/v1/search` |
| Forecast | `https://api.open-meteo.com/v1/forecast` |

### Dynamic field selection

The Open-Meteo request is built **dynamically** from the union of `required_fields` across all loaded SOPs (see [`app/weather/field_requirements.py`](app/weather/field_requirements.py)).

Adding a new SOP with new `required_fields` automatically propagates to the weather request — **no code changes needed** (provided the fields are supported by Open-Meteo).

---

## 7. WeatherFacts

Defined in [`app/weather/models.py`](app/weather/models.py).

```python
class WeatherFacts(BaseModel):
    temperature_2m: Optional[float]        # °C
    apparent_temperature: Optional[float]  # °C
    relative_humidity_2m: Optional[float]  # %
    precipitation: Optional[float]         # mm
    precipitation_probability: Optional[float]  # %
    weathercode: Optional[int]             # WMO code
    visibility: Optional[float]            # meters
    wind_speed_10m: Optional[float]        # km/h
    wind_gusts_10m: Optional[float]        # km/h
    uv_index: Optional[float]
    precipitation_sum_today: Optional[float]    # mm
    precipitation_sum_next_2d: Optional[float]  # mm
```

- All values come **exclusively** from the Open-Meteo API response.
- A `None` value means the field was unavailable.
- A missing field **CANNOT** cause a SOP condition to pass.

---

## 8. SOP Schema

SOPs are stored in [`policies/sops.yaml`](policies/sops.yaml).

```yaml
- id: SOP-XXX
  title: "Human-readable title"
  category: outdoor_exercise          # primary category
  severity: critical                  # low | moderate | high | critical
  match_type: rule                    # rule | score
  overrides: false                    # true = universal emergency override
  priority: 10                        # lower = higher priority in ties
  applies_to_categories:              # categories this SOP applies to
    - outdoor_exercise
  required_fields:                    # Open-Meteo variables needed
    - apparent_temperature
  condition:                          # safe condition AST
    type: all                         # all | any | not | score_gte | score_lt | leaf
    conditions:
      - type: leaf
        field: apparent_temperature
        op: gte                       # gt | gte | lt | lte | eq | ne | in | contains | between
        value: 41
  advice_template: >
    Template text with {apparent_temperature} placeholder.
```

### Condition grammar

| Node type | Description |
|---|---|
| `leaf` | Single field test |
| `all` | Logical AND |
| `any` | Logical OR |
| `not` | Logical NOT |
| `score_gte` | Weighted score ≥ threshold |
| `score_lt` | Weighted score < threshold |

**No `eval()` or `exec()` is used.** All conditions are evaluated by explicit Python functions in [`app/policy/evaluator.py`](app/policy/evaluator.py).

---

## 9. SOP Library

| ID | Title | Category | Severity | Type |
|---|---|---|---|---|
| SOP-001 | Extreme Apparent Heat During Exercise | outdoor_exercise | critical | rule |
| SOP-002 | Heat + Humidity During Strenuous Exercise | outdoor_exercise | high | rule |
| SOP-004 | Poor Visibility During Running or Cycling | outdoor_exercise | high | rule |
| SOP-005 | Thunderstorm During Any Outdoor Activity | all | **critical override** | rule |
| SOP-007 | Strong Wind Gusts During Open-Air Recreation | outdoor_recreation | high | rule |
| SOP-008 | Picnic / Casual Outing Suitability — Favorable | outdoor_recreation | low | **score** |
| SOP-008B | Picnic / Casual Outing Suitability — Marginal | outdoor_recreation | moderate | **score** |
| SOP-010 | Dense Fog During Road Travel | travel | critical | rule |
| SOP-012 | Strong Wind Gusts During Two-Wheeler Travel | travel | high | rule |
| SOP-014 | Child Outdoor Activity During High Heat | vulnerable_groups | high | rule |
| SOP-019 | Sustained Regional Storm / Rain System | all | **critical override** | rule |

### Fuzzy policy (SOP-008 / SOP-008B)

The picnic suitability SOPs use a deterministic weighted score:

| Criterion | Condition | Weight |
|---|---|---|
| Temperature | 18–29°C | 2 |
| Rain probability | < 20% | 2 |
| Wind speed | < 20 km/h | 1 |
| UV index | < 7 | 1 |

- Score ≥ 4 → **SOP-008** (favorable)
- Score < 2 → **SOP-008B** (marginal/poor)
- Score 2–3 → **neither SOP fires** (intentional neutral band — no LLM fills the gap)

---

## 10. Policy Matching

The matcher ([`app/policy/matcher.py`](app/policy/matcher.py)):

1. Loads all SOPs from YAML
2. Filters by `applies_to_categories` (SOPs with `["*"]` apply universally)
3. Evaluates the condition AST against WeatherFacts
4. Returns every matching SOP as a `MatchResult`

Each `MatchResult` includes:

```json
{
  "sop_id": "SOP-012",
  "sop_title": "Strong Wind Gusts During Two-Wheeler Travel",
  "severity": "high",
  "overrides": false,
  "matched_conditions": {
    "wind_gusts_10m": 53.2
  }
}
```

**No LLM call in this step.**

---

## 11. Conflict Resolution

When multiple SOPs match ([`app/policy/selector.py`](app/policy/selector.py)):

1. **Override filter**: If any override SOPs (`overrides: true`) match, only they are eligible to become primary.
2. **Severity**: Among eligible, `critical > high > moderate > low`.
3. **Priority tiebreaker**: Lower `priority` number wins.
4. **Secondary matches**: All non-primary matches are preserved as `secondary_matches`.

**This behavior is deterministic.** No LLM call.

---

## 12. Session Memory

Uses **LangGraph MemorySaver** (in-memory checkpointing):

- Each browser session gets a unique `thread_id`
- The graph persists state across turns within the same thread
- Follow-up queries (e.g., *"what about this evening?"*) inherit location, mode, and group from previous turns
- Memory does **not** persist between independent sessions (in-memory only)

---

## 13. Failure Handling

| Failure | Response |
|---|---|
| Location not resolvable | Honest message: cannot resolve location, cannot proceed |
| Weather API error | Honest message: cannot retrieve live data, cannot recommend |
| No SOP matches | Honest message: no applicable policy found |

**In all failure cases, the LLM is NOT called to invent advice.**

---

## 14. Add-a-Policy Workflow

To add a new safety policy **without changing any application code**:

### Step 1: Open `policies/sops.yaml`

### Step 2: Add a new entry

```yaml
- id: SOP-020
  title: "Extreme Cold During Outdoor Exercise"
  category: outdoor_exercise
  severity: high
  match_type: rule
  overrides: false
  priority: 25
  applies_to_categories:
    - outdoor_exercise
  required_fields:
    - temperature_2m       # <-- must be a supported Open-Meteo variable
  condition:
    type: leaf
    field: temperature_2m
    op: lte
    value: -10
  advice_template: >
    ⚠️ **SOP-020 | Extreme Cold During Outdoor Exercise | HIGH**
    Temperature is {temperature_2m}°C. Conditions are dangerously cold.
    Dress in insulating layers and limit exposure time.
```

### Step 3: Restart the application

```bash
streamlit run frontend/streamlit_app.py
```

### Step 4: Test

Ask: *"Is it safe to run outside in Helsinki today?"*

If the temperature is ≤ −10°C, SOP-020 will fire.

> **Note:** No changes are needed to `graph.py`, the policy matcher, the weather-fetch code, or the LLM code.
>
> **Limitation:** If your new SOP requires an Open-Meteo variable not currently mapped in [`app/weather/field_requirements.py`](app/weather/field_requirements.py), you will need to add that mapping. Only variables already supported by Open-Meteo's API can be used.

---

## 15. Setup

### Prerequisites

- Python 3.11+
- OpenAI API key

### Install

```bash
cd weather-advisory-support-bot
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## 16. Running the Frontend

```bash
streamlit run frontend/streamlit_app.py
```

Opens at `http://localhost:8501`

**Example queries:**
- "Is it safe to cycle in Mumbai today?"
- "Should I take my child to the park in Delhi?"
- "Is today good for a picnic in London?"
- "Can I ride my scooter this afternoon in Chennai?"
- "Should I go for a run in Bhopal?"

---

## 17. Running Tests

```bash
# All unit tests (no API key required for most)
pytest tests/ -v

# Skip live tests (those that make real API calls)
pytest tests/ -v -m "not live"

# Run only policy tests (fully deterministic)
pytest tests/test_policy_loader.py tests/test_policy_evaluator.py tests/test_policy_selector.py -v

# Run adversarial tests
pytest tests/test_adversarial.py -v

# Run graph integration tests (mocked, no API key needed)
pytest tests/test_graph.py -v -m "not integration"
```

---

## 18. Running Evaluations

```bash
# Requires OPENAI_API_KEY for intent-parse and live cases
python evals/run_evals.py
```

Results are written to [`evals/results.md`](evals/results.md).

### Evaluation Cases

| ID | Name | Type |
|---|---|---|
| EVAL-001 | Clear SOP Match — Thunderstorm Override | mocked |
| EVAL-002 | Clear SOP Match — Extreme Heat | mocked |
| EVAL-003 | Paraphrase — Cycling Query | intent_parse |
| EVAL-004 | Paraphrase — Running/Jog Query | intent_parse |
| EVAL-005 | Live Severe-Weather Integration | live |
| EVAL-006 | No SOP Match — Obscure Activity | mocked |
| EVAL-007 | Weather API Failure | mocked_failure |
| EVAL-008 | Prompt Injection / Adversarial | adversarial |
| EVAL-009 | Multiple SOP Matches — Override Wins | mocked |
| EVAL-010 | Session Follow-Up Memory | session_followup |
| EVAL-011 | New SOP Addition Without Code Changes | new_sop |

---

## 19. Limitations

### SOP-019 Proxy Limitation

SOP-019 ("Sustained Regional Storm / Rain System") is a **proxy indicator** based on Open-Meteo precipitation and wind gusts thresholds. It does **not** directly detect:
- IMD (India Meteorological Department) low-pressure systems
- Official storm warnings or red/orange alerts
- Real-time government emergency notifications

The thresholds were chosen to approximate conditions associated with severe weather systems, but they are not authoritative. **Always check official meteorological alerts** for life-safety decisions.

### Field Mapping Limitation

The dynamic field requirements system maps WeatherFacts fields to Open-Meteo API variables via [`app/weather/field_requirements.py`](app/weather/field_requirements.py). If a new SOP requires an Open-Meteo variable that is not currently in this mapping, the field must be added manually. This is a one-time, low-effort addition but does require a code change.

### Hourly Field Approximation

Fields like `uv_index` and `precipitation_probability` are taken from the **first index** of the hourly forecast array (nearest hour). This is an approximation. For precise time-of-day forecasts, a more sophisticated hourly index selection would be needed.

### In-Memory Session Only

Session memory uses LangGraph's `MemorySaver` (in-memory). Sessions are lost on server restart. For persistent cross-session memory, a database-backed checkpointer would be needed.

### Intent Parsing Edge Cases

The LLM is asked to classify into a closed vocabulary, but edge-case phrasing (e.g., unusual regional dialect, highly ambiguous queries) may produce incorrect classifications. The Pydantic validator rejects out-of-vocabulary values, so the failure mode is a missing category (→ no-match response) rather than hallucination.

### No Authentication

The frontend has no authentication. All users on the same server share a process. This is appropriate for a take-home demo but not for production.

---

## Design Philosophy

> *"The LLM interprets language but does not determine safety policy."*
>
> *"Weather numbers originate only from validated WeatherFacts."*
>
> *"Policy decisions are deterministic."*
>
> *"Adding a policy does not require changes to graph control flow or weather-fetch code when its required fields are supported by Open-Meteo."*
