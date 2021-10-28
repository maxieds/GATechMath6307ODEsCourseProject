#### AddPythonPathToBashConfig.sh: Adds the paths of the local database resources 
####                               Python3 libraries to the user's local Bash shell 
####                               configuration on Linux and MacOS.
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.06.11

#!/bin/bash

OS_DESC=`uname -s`
READLINK=`which readlink`
if [[ "$OS_DESC" == "Darwin" ]]; then
     READLINK=`which greadlink`
fi

BASHRC_FILE_BASE=`$READLINK -f ~/`
if [[ "$OS_DESC" == "Darwin" ]]; then
     BASHRC_FILE=$BASHRC_FILE_BASE/.bash_profile
else
     BASHRC_FILE=$BASHRC_FILE_BASE/.bashrc
fi
if [ ! -f $BASHRC_FILE ]; then
     echo -e "ERROR: Could not locate the user's Bash configuration file ($BASHRC_FILE)."
     exit -1
fi

## Run this from within the UtilityScripts directory: 
LOCAL_ROOT_PATH=`$READLINK -f ..`
PYTHON_PATH_APPEND_DIR_LIST=( PythonODEBaseLibrary )
PYTHON_PATH_APPEND=$LOCAL_ROOT_PATH
for dirPath in $PYTHON_PATH_APPEND_DIR_LIST; do
     PYTHON_PATH_APPEND=$PYTHON_PATH_APPEND:$LOCAL_ROOT_PATH/$dirPath
done

DATESTAMP=`date +"%Y-%m-%d @@ %H:%M:%S"`

BASHRC_APPEND_LINES="\n\n"
BASHRC_APPEND_LINES+="#### --> Math6307 Course Project Libraries:\n"
BASHRC_APPEND_LINES+="#### --> Added on $DATESTAMP\n"
BASHRC_APPEND_LINES+="unset PYTHONPATH\n"
BASHRC_APPEND_LINES+="if [[ \"\$PYTHONPATH\" == \"\" ]]; then\n"
BASHRC_APPEND_LINES+="\texport PYTHONPATH=$PYTHON_PATH_APPEND\n"
BASHRC_APPEND_LINES+="else\n"
BASHRC_APPEND_LINES+="\texport PYTHONPATH=\$PYTHONPATH:$PYTHON_PATH_APPEND\n"
BASHRC_APPEND_LINES+="fi\n"
BASHRC_APPEND_LINES+="#### <-- end Course Project Libraries\n"
BASHRC_APPEND_LINES+="\n"

echo -e "$BASHRC_APPEND_LINES" >> $BASHRC_FILE
echo -e "Appended the following lines to $BASHRC_FILE:"
echo -e "$BASH_APPEND_LINES\n"
echo -e "!!! Remember to run `source $BASHRC_FILE` before moving to the examples !!!\n"

exit 0
