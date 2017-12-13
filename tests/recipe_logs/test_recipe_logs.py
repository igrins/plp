import pytest

from igrins.igrins_libs.recipes import RecipeLog

recipe_content = """
OBJNAME, OBJTYPE, GROUP1, GROUP2, EXPTIME, RECIPE, OBSIDS, FRAMETYPES
# Avaiable recipes : FLAT, THAR, SKY, A0V_AB, A0V_ONOFF, STELLAR_AB, STELLAR_ONOFF, EXTENDED_AB, EXTENDED_ONOFF
, DARK, 1, 1, 30.000000, DEFAULT, 1 2 3 4 5, ON ON ON ON ON
, FLAT, 1, 1, 30.000000, FLAT, 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25, OFF OFF OFF OFF OFF OFF OFF OFF OFF OFF ON ON ON ON ON ON ON ON ON ON
HD1561, STD, 1, 1, 300.000000, A0V_AB, 41 42 43 44, A B B A
HD3765, TAR, 1, 1, 150.000000, STELLAR_AB, 45 46 47 48, A B B A
HD19305, TAR, 49a, 53, 180.000000, STELLAR_AB, 49 50 51 52, A B B A
HIP16322, STD, 1, 1, 150.000000, A0V_AB, 53 54 55 56, A B B A
TTau, TAR, 1, 1, 180.000000, STELLAR_AB|SKY_AB, 57 58 59 60, A B B A
HIP18717, STD, 1, 1, 300.000000, A0V_AB, 61 62 63 64, A B B A
, DARK, 1, 1, 30.000000, DEFAULT, 183 184 185 186 187, ON ON ON ON ON
"""


recipe_content_dup_error = """
OBJNAME, OBJTYPE, GROUP1, GROUP2, EXPTIME, RECIPE, OBSIDS, FRAMETYPES
# Avaiable recipes : FLAT, THAR, SKY, A0V_AB, A0V_ONOFF, STELLAR_AB, STELLAR_ONOFF, EXTENDED_AB, EXTENDED_ONOFF
HD3765, TAR, 49a, 1, 150.000000, STELLAR_AB, 45 46 47 48, A B B A
HD19305, TAR, 49a, 53, 180.000000, STELLAR_AB, 49 50 51 52, A B B A
"""


recipe_content_check_error = """
OBJNAME, OBJTYPE, GROUP1, GROUP2, EXPTIME, RECIPE, OBSIDS, FRAMETYPES
# Avaiable recipes : FLAT, THAR, SKY, A0V_AB, A0V_ONOFF, STELLAR_AB, STELLAR_ONOFF, EXTENDED_AB, EXTENDED_ONOFF
HD1561, STD, 40, 1, 300.000000, A0V_AB, 41 42 43 44, A B B A
"""


@pytest.fixture(scope='session')
def recipe_file1(tmpdir_factory):
    fn = tmpdir_factory.mktemp('recipe_log').join('test.recipes')
    open(str(fn), "w").write(recipe_content)
    return fn


@pytest.fixture(scope='session')
def recipe_file_dup_err(tmpdir_factory):
    fn = tmpdir_factory.mktemp('recipe_log').join('test_dup_err.recipes')
    open(str(fn), "w").write(recipe_content_dup_error)
    return fn


@pytest.fixture(scope='session')
def recipe_file_check_err(tmpdir_factory):
    fn = tmpdir_factory.mktemp('recipe_log').join('test_check_err.log')
    open(str(fn), "w").write(recipe_content_check_error)
    return fn


def test_load(recipe_file1):
    # r = load_recipe(fn)
    r = RecipeLog(recipe_file1)

    assert r.iloc[1]["starting_obsid"] == "6"


def test_group1(recipe_file1):

    r = RecipeLog(recipe_file1)
    assert r.iloc[1]["group1"] == "6"
    assert r.iloc[4]["group1"] == "49a"


def test_group2(recipe_file1):

    r = RecipeLog(recipe_file1)
    assert r.iloc[0]["group2"] == "1"
    assert r.iloc[4]["group2"] == "53"


def test_check_err(recipe_file_check_err):
    with pytest.raises(ValueError):

        RecipeLog(recipe_file_check_err)


def test_dup_err(recipe_file_dup_err):
    with pytest.raises(ValueError):

        RecipeLog(recipe_file_dup_err)


def test_multiple_recipe(recipe_file1):

    r = RecipeLog(recipe_file1)
    r1 = r._select_recipe_fnmatch("SKY_*")
    assert len(r1) == 1
    assert r1.iloc[0]["group1"] == "57"


def test_multiple_recipe2(recipe_file1):

    r = RecipeLog(recipe_file1)
    r1 = r.subset(recipe_fnmatch="SKY_*")
    assert len(r1) == 1
    assert r1.iloc[0]["group1"] == "57"


def test_subset(recipe_file1):

    r = RecipeLog(recipe_file1)
    r1 = r.subset(obstype="DARK")
    assert len(r1) == 2
    assert r1.iloc[0]["group1"] == "1"
    assert r1.iloc[1]["group1"] == "183"
