from importlib.resources import files
import shutil
from pathlib import Path

def init_master_calib():

    srcdir = files("igrins") / "master_calib"
    destdir = Path("master_calib")

    if destdir.exists():
        print(f"{destdir} already exists.")
        return

    # destdir.mkdir(parents=True)
    shutil.copytree(srcdir, destdir)

    print(f"created {destdir} directory and populated master calibration files.")

    # now copy recipe.config
    srcdir = files("igrins") / "recipe_config" / "recipe.config"
    destdir = Path(".") / "recipe.config"

    if destdir.exists():
        print(f"{destdir} already exists. We skip copying {destdir}")
        return

    shutil.copy(srcdir, destdir)

    print(f"created {destdir}.")
