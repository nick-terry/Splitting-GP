set source_dir=C:\Users\pnter\Documents\GitHub\GP-Regression-Research\

set file_list=(LocalGP.py SplittingLocalGP.py SplittingLocalGPTests.py UtilityFunctions.py varBoundFunctions.py LocalGP_Kfold_Crossvalidation.py LocalGPTests.py LocalVariationalGPTests.py MemoryHelper.py)

set dest_path=pnterry@mox.hyak.uw.edu://gscratch/choe/pnterry/gp_code

rmdir /s /q "%source_dir%FilesToHyak"
mkdir "%source_dir%FilesToHyak"

for %%i in %file_list% do (

	xcopy %source_dir%%%i %source_dir%FilesToHyak
)

"C:\Program Files\7-Zip\7z.exe" a -ttar "%source_dir%FilesToHyak.tar" "%source_dir%FilesToHyak\"
pscp "%source_dir%FilesToHyak.tar" "%dest_path%"
rm "%source_dir%FilesToHyak.tar"