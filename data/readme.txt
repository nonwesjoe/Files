1.This folder is for you to put datasets
/data
--/fish-classes
--/cat-classes
--/cow-classes
etc...

2.In this directory the 'getdata.py' is a module used to create dataset for pytorch.
 how to use?
///
from getdata import Get_data

get=Get_data(batch_size=64,size=224)
trainloader,valloader=get.get_fish()
///

