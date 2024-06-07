# wgetall.sh
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
