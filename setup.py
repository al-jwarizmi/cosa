from setuptools import setup

setup(name='co_sa',
    version='0.1',
    description='Computer Optic Semantics',
    packages=['cosa'],
    author = 'Alfredo Lozano',
    author_email = 'lozanoa94@gmail.com',
    install_requires=[
   'scipy',
   'scikit-image',
   'scikit-learn',
   'numpy'
   ],
   zip_safe=False)