import abc
import argparse
import inspect
from argparse import ArgumentParser

from pricePrediction import config


class ArgParseable(abc.ABC):

    @property
    @abc.abstractmethod
    def DESIRED_PARAMS_TO_ASK(self): #TOOD: Properties always apply to instances, not classes. to be modified
        raise NotImplementedError()

    @classmethod
    def fromTypeStrToType(cls, strType: str):
        if strType == "str":
            return str
        elif strType == "float":
            return float
        elif strType == "int":
            return int
        elif strType == "bool":
            return bool
        else:
            raise ValueError("Parser not recognized type: "+str(strType))

    @classmethod
    def fromDefaultStrToValue(cls, argname:str, typeFun:type, defaultStr: str): #Currently not used
        '''
        TO BE USED if  defaults in the docstring E.g.    :param int batch_size: Batch size. defaults to %(config.BATCH_SIZE)s

        :param argname:
        :param typeFun:
        :param defaultStr:
        :return:
        '''

        if defaultStr is None or defaultStr == "None":
            return None
        return typeFun(defaultStr)

    @classmethod
    def getArgsForParser(cls):
        from docstring_parser import parse
        docstring = cls.__init__.__doc__
        config_vars = vars(config)
        config_vars = { "config."+key: val for key,val in config_vars.items() }
        docstring = docstring%config_vars
        docstring = parse(docstring)

        signature = inspect.signature(cls.__init__)
        name_to_default = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        params = []
        for elem in docstring.params:
            if elem.arg_name in cls.DESIRED_PARAMS_TO_ASK:
                typeFun = cls.fromTypeStrToType(elem.type_name)
                # print(elem.default, elem.description)
                ### default = cls.fromDefaultStrToValue(elem.arg_name, typeFun, elem.default)
                default = name_to_default[elem.arg_name]
                params.append((elem.arg_name, typeFun, default, elem.description))
        return params

    # @classmethod
    # def _addParamToArgParse(cls, parser:ArgumentParser, paramTuple):
    #     name, typeFun, default, help= paramTuple
    #     if typeFun == bool:
    #         if default is True:
    #             name = "no_"+name
    #             help = "Deactivate "+help
    #         parser.add_argument("--" + name, help=help+" Default=%(default)s", action="store_true")
    #     else:
    #         parser.add_argument("--"+name, type=typeFun, help=help+" Default=%(default)s", default=default)
    #     return parser

    @classmethod
    def addParamsToArgParse(cls, parser: ArgumentParser):
        '''

        :param parser: Argparser to which methods will be added
        :return: the input parser but with added arguments
        '''
        for paramTuple in cls.getArgsForParser():
            name, typeFun, default, help= paramTuple
            if typeFun == bool:
                if default is True:
                    action="store_false"
                else:
                    action="store_true"
                help += " Action: "+action
                parser.add_argument("--" + name, help=help, action=action)
            else:
                parser.add_argument("--"+name, type=typeFun, help=help+" Default=%(default)s", default=default)
        return parser



class MyArgParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args=args, namespace=namespace)
        arg_groups = {}
        for group in self._action_groups:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            arg_groups[group.title] = group_dict

        return arg_groups