from igrins import get_caldb

def main():

    obsdate, band = "20170215", "K"
    config_file = "recipe.config"

    caldb = get_caldb(obsdate, band, config_file)

    _ = caldb.get_all_products_for_db("flat_on")
    print(_)

    v = caldb.load("flat_on", "FLATCENTROID_JSON")

if __name__ == '__main__':
    main()

