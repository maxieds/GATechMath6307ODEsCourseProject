#### ArgumentParser.py
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.06.11

import sys
import argparse
from argparse import ArgumentParser
from ScriptConfig import *

class NumericalODEArgumentParser(ArgumentParser):

    DEFAULT_PROGRAM_DESCRIPTION = None
    DEFAULT_PROGRAM_EPILOG = None

    BOOLEAN_SHORTHAND_DICT = dict([ 
        ("ON", True), 
        ("OFF", False), 
        ("YES", True), 
        ("NO", False), 
        ("Y", True), 
        ("N", False), 
        ("0", False), 
        ("1", True), 
        ("True", True), 
        ("False", False), 
        ("TRUE", True), 
        ("FALSE", False)
    ])

    def __init__(self, 
                 prog=sys.argv[0], 
                 description=DEFAULT_PROGRAM_DESCRIPTION,
                 epilog=DEFAULT_PROGRAM_EPILOG,
                 argument_default=argparse.SUPPRESS,
                 add_help=True):
        super().__init__(prog=prog,
                         description=description,
                         epilog=epilog,
                         argument_default=argument_default,
                         add_help=add_help)

    def appendDefaultArguments(self):
        self.add_argument('-c', '--terminal-color', 
                          type=str,
                          default="True",
                          choices=self.BOOLEAN_SHORTHAND_DICT.keys(),
                          help="Suppress printing of ANSI escape sequences in terminal output, i.e., color formatted text.",
                          dest="_terminal_color_str")
        self.add_argument('-q', '--quiet', 
                          action="store_true",
                          default=False,
                          help="Suppress printing of all output except fatal exceptions and errors.",
                          dest="_quiet")
        self.add_argument('-v', '--verbose', 
                          action="store_true",
                          default=False,
                          help="Print more detailed output to console.",
                          dest="_verbose")
        self.add_argument('-vv', '--vverbose',
                          action="store_true",
                          default=False,
                          help="Print the most detailed standard output to console.",
                          dest="_vverbose")
        self.add_argument('-d', '--debug', 
                          action="store_true",
                          default=False,
                          help="Turn printing of debugging-only messages ON/OFF by default.",
                          dest="_debugging")

    def parseArgs(self, args = sys.argv[1:], apply_defaults = True):
        parsedArgs = self.parse_args(args)
        if apply_defaults:
            self.applyDefaultVariableTranslations(parsedArgs)
        return parsedArgs
    
    def applyDefaultVariableTranslations(self, argParserResult):
        ScriptConfig.COLOR_PRINTING = self.BOOLEAN_SHORTHAND_DICT[argParserResult._terminal_color_str]
        ScriptConfig.QUIET = argParserResult._quiet
        ScriptConfig.VERBOSE = argParserResult._verbose or argParserResult._vverbose
        ScriptConfig.VVERBOSE = argParserResult._vverbose
        ScriptConfig.DEBUGGING = argParserResult._debugging
