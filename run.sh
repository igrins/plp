#!/bin/bash
#Simple script to fully run IGRINS PLP on any UTDATE
echo "Enter UTDATE:" #Prompt user to enter date
read UTDATE #Get date from user
#Clean up previous files, if any
rm -r outdata/$UTDATE
rm -r calib/secondary/$UTDATE
rm -r calib/primary/$UTDATE
#Run PLP

# python ./igr_pipe.py clean-pattern-setup $UTDATE #Setup files needed for flexure correction



python ./igr_pipe.py flat $UTDATE #Process flat
python ./igr_pipe.py register-sky $UTDATE #First pass at wavelength solution using sky frame (incase THAR lamps don't exist)
python ./igr_pipe.py flexure-setup $UTDATE #Setup files needed for flexure correction
#python ./igr_pipe.py thar $UTDATE #First pass at wavelength solution using ThAr lamps, looks like current PLP version doesn't even use THAR lamp frames
python ./igr_pipe.py wvlsol-sky $UTDATE #Second pass at wavelength solution using OH sky emission
python ./igr_pipe.py a0v-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics #Reduce A0V standard star data
python ./igr_pipe.py a0v-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics
python ./igr_pipe.py tell-wvsol $UTDATE #Use telluric absorption in A0V std star to fine tune wavelength solution
python ./igr_pipe.py stellar-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics #Reduce stellar sources nod on slit
python ./igr_pipe.py stellar-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics #Reduce stellar sources nod off slit
python ./igr_pipe.py extended-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics #Reduce extended sources nod on slit
python ./igr_pipe.py extended-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics #Reduce extended sources not off slit
python ./igr_pipe.py divide-a0v $UTDATE #Reduce stellar sources nod off slit
#python igr_pipe.py plot-spec $UTDATE --html-output #Make and publish HTML preview
#python igr_pipe.py publish-html $UTDATE #Ditto

echo "Done running"
echo $UTDATE

