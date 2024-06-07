# wgetall.sh
PWD0=$(pwd)
cd data
# mkdir -p corpus_thinkpython
# cd corpus_thinkpython

DOMAIN=greenteapress.com
URL=https://www.greenteapress.com/thinkpython/thinkCSpy/html/

wget \
     --recursive \
     --level 5 \
     --no-clobber \
     --page-requisites \
     --adjust-extension \
     --span-hosts \
     --convert-links \
     --restrict-file-names=windows \
     --domains $DOMAIN \
     --no-parent \
         $URL

rm -rf greenteapress.com/
mv www.greenteapress.com/thinkpython corpus_thinkpython
rm -rf www.greenteapress.com/
cd corpus_thinkpython/
mv thinkCSpy/* .
rm -rf thinkCSpy
cd ..
git add corpus_thinkpython/
git commit -am 'scraped thinkpython HTML textbook from GreenTeaPress.com'
cd $PWD0