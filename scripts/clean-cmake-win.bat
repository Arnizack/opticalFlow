cd ..
git clean -Xdf -n --exclude="!CMakeSettings.py !"

SET /P input="Do you want to delete the files [Y,N]?"

echo %input%
IF %input%==Y git clean -Xdf 
