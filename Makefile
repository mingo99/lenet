run:
	python lenettrain.py

clean: clean_cp clean_log

clean_cp:
	rm -rf ./checkpoint/*

clean_log:
	rm -rf ./log/*