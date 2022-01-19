from setuptools import setup

setup(name='cosa',
    version='0.0.3',
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
   zip_safe=False,
)
