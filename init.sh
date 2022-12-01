#!/usr/bin/bash

# touch __init__.py
# touch models/__init__.py

for item in models/*; do
	if [[ -d $item ]]
	then
		cd $item
		# touch __init__.py
		ln -sfv ../../share.py share.py
		cd -
	fi
done

cd scripts/
ln -sfv ../share.py share.py
cd -