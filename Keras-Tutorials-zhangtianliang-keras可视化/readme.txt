该教程在知乎：https://zhuanlan.zhihu.com/p/23748037?refer=crane

keras安装plot环境：
Microsoft Windows [版本 10.0.14393]
(c) 2016 Microsoft Corporation。保留所有权利。

C:\Users\Catherine>pip install graphviz
Collecting graphviz
Exception:
Traceback (most recent call last):
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\basecommand.py", line 215, in main
    status = self.run(options, args)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\commands\install.py", line 335, in run
    wb.build(autobuilding=True)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\wheel.py", line 749, in build
    self.requirement_set.prepare_files(self.finder)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\req\req_set.py", line 380, in prepare_files
    ignore_dependencies=self.ignore_dependencies))
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\req\req_set.py", line 554, in _prepare_file
    require_hashes
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\req\req_install.py", line 278, in populate_link
    self.link = finder.find_requirement(self, upgrade)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 465, in find_requirement
    all_candidates = self.find_all_candidates(req.name)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 423, in find_all_candidates
    for page in self._get_pages(url_locations, project_name):
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 568, in _get_pages
    page = self._get_page(location)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 683, in _get_page
    return HTMLPage.get_page(link, session=self.session)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 811, in get_page
    inst = cls(resp.content, resp.url, resp.headers)
  File "D:\DeepLearning\Anaconda\lib\site-packages\pip\index.py", line 731, in __init__
    namespaceHTMLElements=False,
TypeError: parse() got an unexpected keyword argument 'transport_encoding'

C:\Users\Catherine>conda install pip
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.4.6
  latest version: 4.4.7

Please update conda by running

    $ conda update -n base conda



## Package Plan ##

  environment location: D:\DeepLearning\Anaconda

  added / updated specs:
    - pip


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    pip-9.0.1                  |   py36h226ae91_4         2.5 MB

The following packages will be UPDATED:

    pip: 9.0.1-py36hadba87b_3 --> 9.0.1-py36h226ae91_4

Proceed ([y]/n)? y


Downloading and Extracting Packages
pip 9.0.1: #################################################################################################### | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

C:\Users\Catherine>pip install graphviz
Collecting graphviz
  Downloading graphviz-0.8.2-py2.py3-none-any.whl
Installing collected packages: graphviz
Successfully installed graphviz-0.8.2

C:\Users\Catherine>pip install pydot
Collecting pydot
  Downloading pydot-1.2.4.tar.gz (132kB)
    100% |████████████████████████████████| 133kB 209kB/s
Requirement already satisfied: pyparsing>=2.1.4 in d:\deeplearning\anaconda\lib\site-packages (from pydot)
Building wheels for collected packages: pydot
  Running setup.py bdist_wheel for pydot ... done
  Stored in directory: C:\Users\Catherine\AppData\Local\pip\Cache\wheels\62\48\83\42bc8712cb5f9bb93b8f3804e84b31024046981097729ff44e
Successfully built pydot
Installing collected packages: pydot
Successfully installed pydot-1.2.4

C:\Users\Catherine>pip install pydot_ng
Collecting pydot_ng
  Downloading pydot_ng-1.0.0.zip
Requirement already satisfied: pyparsing>=2.0.1 in d:\deeplearning\anaconda\lib\site-packages (from pydot_ng)
Building wheels for collected packages: pydot-ng
  Running setup.py bdist_wheel for pydot-ng ... done
  Stored in directory: C:\Users\Catherine\AppData\Local\pip\Cache\wheels\4f\09\d5\f96fd2578831e1b9021c634f057ab5306a3e4287efa800de29
Successfully built pydot-ng
Installing collected packages: pydot-ng
Successfully installed pydot-ng-1.0.0



在conda中重新安装pip，conda install pip
先安装graphcviz
在安装graphcviz.msi
在安装pyparsing
在安装pydot

最后，重启电脑，ok


