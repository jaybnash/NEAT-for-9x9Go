Script that trains a NEAT algorithm to play 9x9 Go. The NEAT parameters are controlled by the text file.

neat_selfplay trains to beat itself while neat_v_baseline tries to beat a baseline agent created by pgx

Note that neat_v_baseline has a very, very bad fitness function and should be changed if possible. If not, using selfplay to for many generations before using neat_v_baseline might allow the bot to be good enough to fight the baseline with fitness only granted based on win/lose, which is how selfplay works
