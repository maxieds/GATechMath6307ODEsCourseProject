#### Utils.py
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.10.07

from datetime import datetime
from ScriptConfig import *
import cmath
from random import seed, randint
from sage.all import *

class Utils:

    @staticmethod
    def IsStringType(x):
        return type(x) == str or type(x) == bytearray

    @staticmethod
    def IsIntType(x): 
        return type(x) == int 

    @staticmethod
    def IsFloatType(x):
        return type(x) == float

    @staticmethod
    def IsListType(x):
        return type(x) == list or type(x) == tuple

    @staticmethod
    def Python3BytesToStringConvert(s):
        """Simplify PythonV3 string versus bytes type handling."""
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return str(s.decode('utf-8').encode('utf-8'))[2:-1]
        return s

    @staticmethod
    def GetTimestamp():
        """Return a timestamp."""
        return datetime.now().strftime("%Y-%m-%d-%H%M%S")

    @staticmethod
    def ReadTextFileContents(filename):
        if not exists(filename) or isdir(filename):
            raise FileNotFoundError(filename)
        with open(filename, encoding = 'utf8', errors = 'ignore') as inputFile:
            return inputFile.read()

    @staticmethod
    def WriteToFile(filename, content, overwrite_if_exists = False):
        if exists(filename) and not overwrite_if_exists:
            raise FileExistsError(filename)
        with open(filename, 'w') as outfile:
            outfile.write(content)

    @staticmethod
    def Python3ComplexNorm(zz):
        xx, yy = zz.real, zz.imag
        return cmath.sqrt(xx*xx + yy*yy)

    @staticmethod
    def SageMathNorm(zz):
        zz = n(zz)
        if isinstance(zz, complex):
            return Utils.Python3ComplexNorm(zz)
        elif not zz.is_real() and hasattr(zz, 'norm'):
            return norm(zz)
        return zz

    @staticmethod
    def GetShortObjectHash(objectsLst, shortHashLength=6, deterministic=True):
        if not isinstance(shortHashLength, int) or shortHashLength <= 0:
            raise ValueError
        if len(objectsLst) == 0:
            return "0" * shortHashLength
        if not deterministic:
            seed(int(hash(datetime.now())))
        objHashLst = [ int(Utils.GetShortObjectHash(obj, 32), 16) if isinstance(obj, list) else hash(obj) for obj in list(objectsLst) ]
        modFullObjListHash = objHashLst[0]
        for (hoIdx, curHash) in enumerate(objHashLst[1:]):
            if deterministic:
                termMask = hoIdx | (hoIdx << 6) | (hoIdx << 12) | (hoIdx << 18)
            else:
                termMask = randint(0, 2**32 - 1)
            modFullObjListHash ^= curHash & termMask
        hashStrFmt = "%0" + ("%d" % shortHashLength) + "u"
        truncFullHash = (hashStrFmt % modFullObjListHash)[1:shortHashLength+1]
        return truncFullHash.upper()

    class PPrint:

        ## """ ANSI color codes """
        BLACK = "\033[0;30m"
        RED = "\033[0;31m"
        GREEN = "\033[0;32m"
        BROWN = "\033[0;33m"
        BLUE = "\033[0;34m"
        PURPLE = "\033[0;35m"
        CYAN = "\033[0;36m"
        LIGHT_GRAY = "\033[0;37m"
        DARK_GRAY = "\033[1;30m"
        LIGHT_RED = "\033[1;31m"
        LIGHT_GREEN = "\033[1;32m"
        YELLOW = "\033[1;33m"
        LIGHT_BLUE = "\033[1;34m"
        LIGHT_PURPLE = "\033[1;35m"
        LIGHT_CYAN = "\033[1;36m"
        LIGHT_WHITE = "\033[1;37m"
        BOLD = "\033[1m"
        FAINT = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        NEGATIVE = "\033[7m"
        CROSSED = "\033[9m"
        END = "\033[0m"

        @staticmethod
        def PrintError(emsg):
            if not ScriptConfig.QUIET and ScriptConfig.COLOR_PRINTING:
                 print(PPrint.BOLD + PPrint.LIGHT_RED + " >> " +
                       PPrint.RED + "ERROR: " + PPrint.LIGHT_BLUE + emsg + PPrint.END)
            elif not ScriptConfig.QUIET:
                print(" >> ERROR: " + emsg)

        @staticmethod
        def PrintWarning(emsg, headerPrefix = "WARNING"):
            if not ScriptConfig.QUIET and ScriptConfig.COLOR_PRINTING:
                 print(PPrint.BOLD + PPrint.LIGHT_GRAY + " >> " +
                       PPrint.YELLOW + "{0}: ".format(headerPrefix) + PPrint.PURPLE + emsg + PPrint.END)
            elif not ScriptConfig.QUIET:
                 print(" >> {0}: {1}".format(headerPrefix, emsg))

        @staticmethod
        def PrintInfo(emsg):
            if not ScriptConfig.QUIET and ScriptConfig.COLOR_PRINTING:
                print(PPrint.BOLD + PPrint.LIGHT_GRAY + " >> " +
                      PPrint.LIGHT_GREEN + "INFO: " + PPrint.LIGHT_BLUE + emsg + PPrint.END)
            elif not ScriptConfig.QUIET:
                print(" >> INFO: " + emsg)

        @staticmethod
        def PrintDebug(emsg):
            if not ScriptConfig.QUIET and ScriptConfig.DEBUGGING and ScriptConfig.COLOR_PRINTING:
                print(PPrint.BOLD + PPrint.LIGHT_GRAY + " >> " +
                      PPrint.LIGHT_GREEN + "DEBUGGING: " +
                      PPrint.LIGHT_BLUE + emsg + PPrint.END)
            elif not ScriptConfig.QUIET and ScriptConfig.DEBUGGING:
                print(" >> DEBUGGING: " + emsg)
