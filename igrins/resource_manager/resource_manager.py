from resource_context import ResourceContextStack
# from item_convert import ItemConverterBase


class ResourceStack(object):
    """
    categories:
      config
      refdata (or master cal data)
      resource
      items
    """

    def __init__(self, resource_spec, storage, resource_db,
                 item_converter_class=None,
                 qa_generator=None):

        self._resource_spec = resource_spec

        if item_converter_class is None:
            self.storage = storage
        else:
            self.storage = item_converter_class(storage)

        self.resource_db = resource_db

        self.qa_generator = qa_generator

        self.context_stack = ResourceContextStack(self.storage)

    # CONTEXT

    def new_context(self, context_name):
        self.context_stack.new_context(context_name)

    def close_context(self, context_name=None):
        if context_name is not None:
            assert self.context_stack.current.name == context_name

        self.context_stack.close_context()

    # # conversion
    # def get_convert_to(self, item_desc, item_type):
    #     item_type = item_type if item_type else \
    #                 self.item_converter.guess_item_type(item_desc)

    #     return self.item_converter.get_to_item(item_type)

    # def get_convert_from(self, item_desc, item_type):
    #     item_type = item_type if item_type else \
    #                 self.item_converter.guess_item_type(item_desc)

    #     return self.item_converter.get_from_item(item_type)

    def get_section_n_fn(self, basename, item_desc, postfix=""):

        (section, tmpl) = item_desc
        fn = tmpl.format(basename=basename, postfix=postfix)

        return section, fn

    def load(self, basename, item_desc, item_type=None, postfix=""):

        section, fn = self.get_section_n_fn(basename, item_desc, postfix)

        d = self.context_stack.load(section, fn, item_type=item_type)
        return d

    def store(self, basename, item_desc, data, item_type=None,
              postfix=""):

        section, fn = self.get_section_n_fn(basename, item_desc, postfix)

        self.context_stack.store(section, fn, data, item_type=item_type)

    # RESOURCE

    def query_resource_for(self, basename, resource_type, postfix=""):
        """
        query resource from the given basename
        """

        _ = self.resource_db.query_resource_for(basename, resource_type,
                                                postfix=postfix)
        resource_basename, item_desc = _

        return resource_basename, item_desc

    def load_resource_for(self, basename, resource_type,
                          item_type=None, postfix="",
                          resource_postfix=""):
        """
        this load resource from the given master_obsid.
        """

        resource_basename, item_desc = self.query_resource_for(basename,
                                                               resource_type,
                                                               postfix=postfix)

        resource = self.load(resource_basename, item_desc,
                             item_type=item_type, postfix=resource_postfix)

        return resource

    def update_db(self, db_name, basename):
        self.resource_db.update_db(db_name, basename)


class ResourceStackWithBasename(ResourceStack):
    """
    categories:
      config
      refdata (or master cal data)
      resource
      items
    """

    def __init__(self, resource_spec, storage, resource_db,
                 basename_helper,
                 item_converter_class=None,
                 qa_generator=None):

        ResourceStack.__init__(self, resource_spec, storage, resource_db,
                               item_converter_class=item_converter_class,
                               qa_generator=qa_generator)

        self.basename_helper = basename_helper

    def get_section_n_fn(self, basename, item_desc, postfix=""):
        basename = self.basename_helper.to_basename(basename)

        return ResourceStack.get_section_n_fn(self, basename, item_desc,
                                              postfix=postfix)

    # RESOURCE

    def query_resource_for(self, basename, resource_type, postfix=""):
        """
        query resource from the given basename
        """

        basename = self.basename_helper.to_basename(basename)

        _ = ResourceStack.query_resource_for(self, basename, resource_type,
                                             postfix=postfix)
        resource_basename, item_desc = _

        resource_basename = self.basename_helper.from_basename(resource_basename)

        return resource_basename, item_desc

    def load_resource_for(self, basename, resource_type,
                          item_type=None, postfix="",
                          resource_postfix=""):
        """
        this load resource from the given master_obsid.
        """

        resource_basename, item_desc = self.query_resource_for(basename,
                                                               resource_type,
                                                               postfix=postfix)

        resource = self.load(resource_basename, item_desc,
                             item_type=item_type, postfix=resource_postfix)

        return resource

    def update_db(self, db_name, basename):
        basename = self.basename_helper.to_basename(basename)

        self.resource_db.update_db(db_name, basename)


if __name__ == "__main__":

    utdate = "20150525"
    band = "H"
    obsids = [52]
    master_obsid = obsids[0]

    from recipe_helper import RecipeHelper
    helper = RecipeHelper("../recipe.config", utdate)

    caldb = helper.get_caldb()
    resource = caldb.load_resource_for((band, master_obsid),
                                       resource_type="aperture_definition")

    basename = caldb.db_query_basename("flat_on", (band, master_obsid))

    resource = caldb.load_item_from(basename,
                                    "FLATCENTROID_SOL_JSON")
