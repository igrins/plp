"""
db_factory: callable object that takes a single argument of db_type and
returns an instance of db, which should define two methods of

def update(self, basename):
    pass

def query(self, basename):
    pass

resource_def : a dict of name, (db_type, item_desc) pairs.
"""


class ResourceDBSet(object):
    def __init__(self, resource_spec,
                 db_factory, resource_def):

        self.resource_spec = resource_spec
        self.resource_def = resource_def

        self.db_factory = db_factory

        self.db_set = {}

    def _get_db(self, db_type):

        if db_type not in self.db_set:
            db = self.db_factory(db_type)

            self.db_set[db_type] = db

        return self.db_set[db_type]

    def update_db(self, db_type, basename):
        db = self._get_db(db_type)
        db.update(basename)

    def query_resource_basename(self, db_type, basename):
        """
        query basename from the given basename
        """

        db = self._get_db(db_type)
        return db.query(basename)

    def query_resource_for(self, basename, resource_type):
        """
        query resource from the given basename
        """

        db_type, item_desc = self.resource_def.get(resource_type,
                                                   resource_type)

        resource_basename = self.query_resource_basename(db_type, basename)

        return resource_basename, item_desc
