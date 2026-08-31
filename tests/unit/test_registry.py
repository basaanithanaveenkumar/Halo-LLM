import pytest

from hale_llm.core.registry import NamedRegistry, VariantRegistry


def test_named_registry_rejects_duplicates():
    reg = NamedRegistry("thing")
    reg.add("a", 1)
    with pytest.raises(ValueError, match="already registered"):
        reg.add("a", 2)


def test_variant_registry_prefers_specific_over_generic():
    reg = VariantRegistry("metric")

    class ArAcc:
        pass

    class GenericAcc:
        pass

    reg.add("acc", ArAcc, variants=("ar",))
    reg.add("acc", GenericAcc, variants=None)

    assert reg.resolve("acc", "ar") is ArAcc
    assert reg.resolve("acc", "diffusion") is GenericAcc


def test_variant_registry_skips_when_no_match():
    reg = VariantRegistry("metric")

    class ArAcc:
        pass

    reg.add("acc", ArAcc, variants=("ar",))
    assert reg.resolve("acc", "diffusion") is None
    assert reg.instantiate(["acc"], "diffusion") == []
