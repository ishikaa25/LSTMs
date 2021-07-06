#1. Clone litbank repo to main directory and run
#2. Clone book-nlp repo inside 'litbank/tagger/' folder
cd litbank/tagger/

for i in `ls ../../Data/Stories`
do echo $i
#Generating tokens from Book-NLP
cd book-nlp/
./runjava novels/BookNLP -doc ../../../Data/Stories/$i -p ../../../Data/Extra/${i%.*} -tok ../../../Data/Tokens/${i%.*}.tokens -f
#Generate tagged file (with EVENTS only)
cd ..
python run_tagger.py --mode predict -i ../../Data/Tokens/${i%.*}.tokens -o ../../Data/Outputs/${i%.*}.tagged
done
#NOTE: Make an empty 'Outputs' folder inside 'Data/' before running this.
