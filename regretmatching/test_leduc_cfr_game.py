from leduc_cfr_game import LeducCFR

def test_which_player():
	assert LeducCFR.which_player([]) == 0
	assert LeducCFR.which_player([11]) == 0
	assert LeducCFR.which_player([11,11]) == 1
	assert LeducCFR.which_player([11,11,1]) == 2
	assert LeducCFR.which_player([11,11,1,2]) == 1
	assert LeducCFR.which_player([11,11,1,2,2]) == 2
	assert LeducCFR.which_player([11,11,1,2,2,1]) == 0
	assert LeducCFR.which_player([11,11,1,2,2,1,12]) == 2

	assert LeducCFR.which_player([11,11,1,2,2,1,12,1]) == 1
	assert LeducCFR.which_player([11,11,1,2,2,1,12,1,2]) == 2
	#assert LeducCFR.which_player([11,11,1,2,2,1,12,1,2,1]) == 2

def test_available_actions():
	assert LeducCFR.available_actions([11,11]) == [1,2]
	assert LeducCFR.available_actions([11,11,1,1,12]) == [1,2]

if __name__ == "__main__":
	test_which_player()
	test_available_actions()