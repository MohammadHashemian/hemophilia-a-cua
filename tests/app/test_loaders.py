# import pytest
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("__pytest__")


def test_costs_data():
    from app.persistence.loaders import load_typed_json, parse_cost_file
    from app.persistence.schemas.costs import CostFile
    from utils.path_utils import get_project_root

    file_path = get_project_root() / "app/data" / "economic.json"

    result = load_typed_json(file_path, parse_cost_file)

    # 1. Type check
    assert isinstance(result, CostFile)

    # 2. Structure checks
    assert len(result.currencies) > 0
    assert len(result.costs) > 0

    # 3. Deep field validation
    first_cost = result.costs[0]
    assert first_cost.item == "factor_viii"
    assert first_cost.pricing.per_unit["IRR"] > 0
    assert first_cost.assumption.iu_per_microgram > 0


def test_clinical_data():
    from app.persistence.loaders import load_typed_json, parse_clinical
    from app.persistence.schemas.clinicals import ClinicalFile
    from utils.path_utils import get_project_root

    file_path = get_project_root() / "app/data" / "clinical.json"

    result = load_typed_json(file_path, parse_clinical)

    # 1. Type check
    assert isinstance(result, ClinicalFile)


def test_utilities_data():
    from app.persistence.loaders import load_typed_json, parse_utilities
    from app.persistence.schemas.utilities import UtilityFile
    from utils.path_utils import get_project_root

    file_path = get_project_root() / "app/data" / "utilities.json"

    result = load_typed_json(file_path, parse_utilities)

    # 1. Type check
    assert isinstance(result, UtilityFile)
    utilities = result.state_utilities
    assert utilities.healthy.mean == 0.94
    assert utilities.intracranial_hemorrhage.mean == 0.15
    assert utilities.non_ich_major_bleeding.mean == 0.55
    assert utilities.mild_arthropathy.uncertainty.sd == 0.13
    assert result.pettersson_utility_thresholds.end_stage_arthropathy == 40


def test_simulation_data():
    from app.persistence.loaders import load_typed_json, parse_simulation
    from app.persistence.schemas.simulation import SimulationFile
    from utils.path_utils import get_project_root

    file_path = get_project_root() / "app/data" / "simulation.json"

    result = load_typed_json(file_path, parse_simulation)

    # 1. Type check
    assert isinstance(result, SimulationFile)


def test_mortality_data():
    from app.persistence.loaders import load_typed_json, parse_mortality
    from app.persistence.schemas.mortality import MortalityFile
    from utils.path_utils import get_project_root

    file_path = get_project_root() / "app/data" / "mortality.json"

    result = load_typed_json(file_path, parse_mortality)

    # 1. Type check
    assert isinstance(result, MortalityFile)


def test_simulation_mortality_source_defaults_to_iran():
    """simulation.json should declare mortality.source and default to iran."""
    from app.persistence.loaders import load_typed_json, parse_simulation
    from utils.path_utils import get_project_root

    sim = load_typed_json(
        get_project_root() / "app/data" / "simulation.json", parse_simulation
    )
    assert sim.mortality.source == "iran"


def test_model_context_loads_iran_mortality_by_default():
    """ModelContext.load() must read simulation.mortality.source and load
    data/mortality_iran.json when the source is 'iran'."""
    from app.persistence.context import ModelContext

    ctx = ModelContext.load()
    assert ctx.simulation.mortality.source == "iran"
    # The Iran table has a markedly different crude rate than the
    # default placeholder (0.0052 vs 0.0128), so this assertion is a
    # strong "the right file got loaded" signal without hard-coding
    # the full table.
    assert abs(ctx.mortality.crude_annual_rate - 0.005227549788845551) < 1e-12
    assert ctx.mortality.age_specific["90+"] > 0.25  # Iran WPP 90+ rate
    # Sanity: age 1-4 in Iran is an order of magnitude lower than the
    # placeholder table (4.3e-4 vs 1.5e-4 — actually quite close, so
    # just confirm it's a small positive number).
    assert 0 < ctx.mortality.age_specific["1-4"] < 1e-3


def test_mortality_source_can_be_overridden(tmp_path, monkeypatch):
    """Setting simulation.mortality.source to 'poland' must load the
    placeholder table, not the Iran one."""
    import json

    from app.persistence.context import ModelContext

    # Write a temp simulation.json that asks for the poland source.
    sim_path = tmp_path / "simulation.json"
    sim_path.write_text(
        json.dumps(
            {
                "environment": {"mode": "development", "seed": 1},
                "discounting": {
                    "enable": False,
                    "cost_rate_annual": 0.058,
                    "utility_rate_annual": 0.05,
                },
                "psa": {"development": 100, "production": 100},
                "mortality": {"source": "poland"},
                "time": {"weeks_per_year": 52},
            }
        )
    )

    # Point ModelContext.PROJECT_ROOT at tmp_path so it reads our file
    # but still resolves relative paths like "data/mortality.json" from
    # the real project root (since those tables aren't copied here).
    real_root = ModelContext.PROJECT_ROOT
    monkeypatch.setattr(ModelContext, "PROJECT_ROOT", real_root)
    monkeypatch.setattr(
        "app.persistence.context.get_project_root", lambda: real_root
    )

    # Patch the loader to use our temp simulation.json.
    import app.persistence.context as ctx_mod
    import app.persistence.loaders as loaders

    orig_load = loaders.load_typed_json

    def patched(path, parser):
        if str(path).endswith("simulation.json"):
            return orig_load(sim_path, parser)
        return orig_load(path, parser)

    monkeypatch.setattr(loaders, "load_typed_json", patched)
    # ctx_mod imported load_typed_json by name, so patch the local binding too.
    monkeypatch.setattr(ctx_mod, "load_typed_json", patched)

    ctx = ModelContext.load()
    assert ctx.simulation.mortality.source == "poland"
    # Poland placeholder crude rate is 0.0128 — distinct from Iran 0.0052.
    assert abs(ctx.mortality.crude_annual_rate - 0.0128) < 1e-12
