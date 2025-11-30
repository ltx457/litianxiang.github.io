function add_path( )

pth = which('add_path.m');

pth = fileparts(pth);

addpath(pth);

rehash toolboxcache;

savepath;

end