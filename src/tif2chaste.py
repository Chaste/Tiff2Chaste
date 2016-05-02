import sys, os
import argparse
import tif2chastelib

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Converts presegmented tif files (from e.g. Seedwater) to CHASTE simulation input files.')
parser.add_argument(	'-mode',
			help='The type of CHASTE output file(s) to produce.',
			choices = ['vertex', 'immersed', 'potts', 'spheres'],
			dest = 'mode',
			metavar='MODE',
			required = True,
			action = 'store',
			type=str)
parser.add_argument(	'-tif',
			help='The input (presegmented) tif image filename.',
			dest = 'infile',
			metavar='INFILE',
			required = True,
			action = 'store',
			type=str)
parser.add_argument(	'-node',
			help='The output CHASTE node filename.',
			dest = 'outnode',
			metavar = 'NODEFILE',
			action = 'store',
			required = True,
			type=str)
parser.add_argument(	'-cell',
			help='The output CHASTE cell filename.',
			dest = 'outcell',
			metavar = 'CELLFILE',
			action = 'store',
			type=str)
parser.add_argument(	'-margin',
			help='immersed, potts and spheres mode only. The distance (in pixels) from the image edge within which cells are culled. Default is 5.',
			dest = 'margin',
			metavar = 'CULLMARGIN',
			default=5,
			action = 'store',
			type=int)
parser.add_argument(	'-scale',
			help='immersed mode only. Rescales the cells about their centroid to allow a greater fluid gap between them. e.g. 0.8 would resize all cells to 80%% of their original size.',
			dest = 'scaling',
			metavar = 'SCALINGFACTOR',
			default=1.0,
			action = 'store',
			type=float)
args = parser.parse_args()

# Check that option combinations make sense
if args.mode == 'vertex' or args.mode == 'immersed' or args.mode == 'potts':
	if args.outcell == None:
		sys.exit("Cell output file name must be specified for vertex, immersed and potts modes. (Use -cell to specify)")
if args.mode == 'spheres' and args.outcell != None:
		sys.exit("spheres mode only outputs a node file. (No need for -cell)")

# Convert the given tif file according to the specified CHASTE output
if args.mode == "vertex":
	nodes, elements = tif2chastelib.tiff2vertex(args.infile)
	tif2chastelib.write_vertex_files(nodes, elements, args.outnode, args.outcell)
elif args.mode == "immersed":
	nodes, elements = tif2chastelib.tiff2immersed(args.infile, margin=args.margin, scaling=args.scaling)
	tif2chastelib.write_immersed_files(nodes, elements, args.outnode, args.outcell)
elif args.mode == "potts":
	nodes, elements = tif2chastelib.tiff2potts(args.infile, margin=args.margin)
	tif2chastelib.write_potts_files(nodes, elements, args.outnode, args.outcell)
elif args.mode == "spheres":
	nodes = tif2chastelib.tiff2spheres(args.infile, margin=args.margin)
	tif2chastelib.write_spheres_files(nodes, args.outnode)
else:
	sys.exit("Unrecognised mode: ", args.mode); # This should never, ever happen
