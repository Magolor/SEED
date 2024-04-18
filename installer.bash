cd ./SeeD
pip install -r requirements.txt
# python -c "from pyheaven import CMD; CMD('pip uninstall seed -y')"
python -c "from pyheaven import CMD; CMD('pip install -e .')"
cd ..
python post_install.py