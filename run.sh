#!/bin/bash
#Simple script to fully run IGRINS PLP on any UTDATE
USER=Default #Name of user running script, will be passed to FTIS headers
VERSION=igrins_v3.1 #Version of PLP being used to run script, will be passed to FTIS headers
SLIT_PROFILE_METHOD=column #Method for determining slit profile for optimal extraction, column=use running median per column across detector (new method, default),  full=use single median profile across entire detector (old method, might be needed for very low SNR targets)

echo "Enter UTDATE:" #Prompt user to enter date
read UTDATE #Get date from user

if ! [[ "$UTDATE" =~ ^[0-9]+$ ]] #Check if UT date is integer
    then
        echo "UTDATE is incorrect format.  Should be YYYYMMDD."
        exit 1 #End script if the UT DATE is wrong
fi

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
python ./igr_pipe.py a0v-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION --slit-profile-method=$SLIT_PROFILE_METHOD #Reduce A0V standard star data
python ./igr_pipe.py a0v-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION --slit-profile-method=$SLIT_PROFILE_METHOD
python ./igr_pipe.py tell-wvsol $UTDATE #Use telluric absorption in A0V std star to fine tune wavelength solution
python ./igr_pipe.py stellar-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION --slit-profile-method=$SLIT_PROFILE_METHOD #Reduce stellar sources nod on slit
python ./igr_pipe.py stellar-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION --slit-profile-method=$SLIT_PROFILE_METHOD #Reduce stellar sources nod off slit
python ./igr_pipe.py extended-ab $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION #Reduce extended sources nod on slit
python ./igr_pipe.py extended-onoff $UTDATE --correct-flexure --height-2dspec=100 --mask-cosmics --user=$USER --version=$VERSION #Reduce extended sources not off slit
python ./igr_pipe.py divide-a0v $UTDATE --user=$USER --version=$VERSION #Reduce stellar sources nod off slit
#python igr_pipe.py plot-spec $UTDATE --html-output #Make and publish HTML preview
#python igr_pipe.py publish-html $UTDATE #Ditto


#Batch rename to Gemini naming convention
for i in $(seq 1 9999) #Loop through all possible obsids
do
	OBSID=$(printf "%04d" $i)
	mv outdata/$UTDATE\/SDCH_$UTDATE\_$OBSID\.spec.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_H.spec.fits  2>/dev/null #Note 2>/dev/null is to silence mv errors when it doesn't find a file
	mv outdata/$UTDATE\/SDCK_$UTDATE\_$OBSID\spec.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_K.spec.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCH_$UTDATE\_$OBSID\.sum.spec.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_H..sum.spec.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCK_$UTDATE\_$OBSID\.sum.spec.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_K..sum.spec.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCH_$UTDATE\_$OBSID\.spec2d.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_H.spec2d.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCK_$UTDATE\_$OBSID\.spec2d.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_K.spec2d.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCH_$UTDATE\_$OBSID\.spec_a0v.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_H.spec_a0v.fits  2>/dev/null
	mv outdata/$UTDATE\/SDCK_$UTDATE\_$OBSID\.spec_a0v.fits  outdata/$UTDATE\/N$UTDATE\S$OBSID\_K.spec_a0v.fits  2>/dev/null
done


echo "Done running"
echo $UTDATE

