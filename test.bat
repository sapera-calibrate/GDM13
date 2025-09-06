@echo off
echo ======================================================================
echo ProTrace Fingerprint Testing - 1.png and 2.png
echo ======================================================================
echo.

echo [STEP 1] Generating fingerprint for 1.png...
python protrace_spn.py gen --image 1.png --image-id img_1 --out 1_fingerprint.json
echo.

echo [STEP 2] Generating fingerprint for 2.png...
python protrace_spn.py gen --image 2.png --image-id img_2 --out 2_fingerprint.json
echo.

echo [STEP 3] Verifying 1.png against its own fingerprint (should pass)...
python protrace_spn.py verify --image 1.png --fingerprint 1_fingerprint.json --out verify_1_self.json
echo.

echo [STEP 4] Verifying 2.png against its own fingerprint (should pass)...
python protrace_spn.py verify --image 2.png --fingerprint 2_fingerprint.json --out verify_2_self.json
echo.

echo [STEP 5] Cross-verification: 1.png against 2.png fingerprint (should fail)...
python protrace_spn.py verify --image 1.png --fingerprint 2_fingerprint.json --out verify_1_vs_2.json
echo.

echo [STEP 6] Cross-verification: 2.png against 1.png fingerprint (should fail)...
python protrace_spn.py verify --image 2.png --fingerprint 1_fingerprint.json --out verify_2_vs_1.json
echo.

echo ======================================================================
echo TEST COMPLETE - Check the generated JSON files for results:
echo   - Fingerprints: 1_fingerprint.json, 2_fingerprint.json
echo   - Self-verify: verify_1_self.json, verify_2_self.json
echo   - Cross-verify: verify_1_vs_2.json, verify_2_vs_1.json
echo ======================================================================
pause
