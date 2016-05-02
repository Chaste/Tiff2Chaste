"""In this file the main mesh classes 'Mesh', 'Element', and 'Node' are defined.
"""
import matplotlib as mpl
mpl.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Mesh():
    """A mesh similar to a mutable vertex mesh in Chaste or an unstructured grid in VTK."""
    def __init__(self, nodes, elements):

        self.nodes = nodes
        self.elements = elements
        self.frame_id_dictionary = {}
        self.global_id_dictionary = {}
         
        self.assign_node_ids_in_order()
        self.build_frame_id_dictionary()
        
    def assign_node_ids_in_order(self):
        """Assign ids to nodes in the order they appear in the node vector
        """
        for counter, node in enumerate(self.nodes):
            node.id = counter

    def build_frame_id_dictionary(self):
        """Index the frame ids in a dictionary.
           Writes empty dictionary if there is at least one frame id that is set to `None'
           in the mesh.
        """
        self.frame_id_dictionary = {}
        for element_index, element in enumerate(self.elements):
            if element.id_in_frame==None:
                self.frame_id_dictionary = {}
                break
            else:
                self.frame_id_dictionary[element.id_in_frame] = element_index 

    def get_element_with_frame_id(self, id_in_frame):
        """Returns the element that has the given id
        
        Parameters
        ----------
        
        id_in_frame : int
            frame id of the element that should be returned
            
        Returns
        -------
        
        element : the element that has the given frame id
        
        Currently, no checking is done as to whether the mapping elements <-> ids is one-to-one
        """
        assert ( id_in_frame in self.frame_id_dictionary )
        return self.elements[self.frame_id_dictionary[id_in_frame]]
    

    def collect_edges(self):
        """Crawl through the elements and collect all edges
        
        Returns
        -------
        
        edges : numpy integer array
            An numpy array where each row contains an edge.
            Each edge is characterised by two integer values, 
            the ID's of the nodes that it contains.
        """
        edge_list = []
        for element in self.elements:
            for index, node in enumerate(element.nodes):
                next_index = (index + 1)%element.get_num_nodes()
                next_node = element.nodes[next_index]
                this_edge = [node.id, next_node.id]
                #the list() command will make an identical copy of the list
                this_edge_reversed = list(this_edge)
                this_edge_reversed.reverse()
                if (this_edge not in edge_list) and\
                    (this_edge_reversed not in edge_list):
                    edge_list.append(this_edge)
        return np.array(edge_list)
    
    def index_global_ids(self):
        """build a lookup table for which element belongs to given global id.
        
        After elements have been assigned global ids we need to run this method in order
        to be able to interrogate the mesh for elements belonging to a global id
        """
        self.global_id_dictionary = {}
        for index, element in enumerate(self.elements):
            if not element.global_id == None:
                self.global_id_dictionary[element.global_id] = index

    def index_frame_ids(self):
        """build a lookup table for which element belongs to a given frame id.
        
        After elements ids or the position of elements in the element vector have 
        changed we need to run this method in order to be able to interrogate the mesh 
        for elements belonging to a frame id
        """
        self.frame_id_dictionary = {}
        for index, element in enumerate(self.elements):
            if not element.id_in_frame == None:
                self.frame_id_dictionary[element.id_in_frame] = index

    def remove_element_with_frame_id(self, frame_id):
        """Removes the element with the given frame id from self.elements.
        Does not alter any nodes or elements, but keeps the internal mapping
        frame_ids <--> location in element vector up to date.
        
        Parameters
        ----------
        
        frame_id : int
            the id of the element that should be removed from the internal list of elements
        """
        
        remaining_elements = []
        
        for element in self.elements:
            if element.id_in_frame != frame_id:
                remaining_elements.append(element)
                
        self.elements = remaining_elements
        self.index_frame_ids()

    def remove_list_of_nodes(self, node_list):
        """Remove the given nodes from the internal list of nodes. Will not alter
        any elements or nodes.
        
        This is a helper function for kill_element_with_frame_id()
        
        Parameters
        ----------
        
        node_list: list of Node instances
            all the nodes that should be removed from self.nodes()
            
        See also
        --------
        
        kill_element_with_frame_id()
        """
        list_of_node_ids = []
        for node in node_list:
            list_of_node_ids.append(node.id)
        
        remaining_nodes = []
        for node in self.nodes:
            if node.id not in list_of_node_ids:
                remaining_nodes.append(node)
                
        self.nodes = remaining_nodes
        
    def get_maximal_node_id(self):
        """Get the largest id of all nodes in the mesh.
        
        Returns
        -------
        
        maximal_node_id : int
            the largest id among all nodes in the mesh
        """
        
        all_node_ids = []

        for node in self.nodes:
            all_node_ids.append(node.id)
        
        max_node_id = np.max(np.array( all_node_ids ))
        
        return max_node_id
    
    def remove_boundary_elements(self):
        """Removes all elements at the outside of the mesh
        
        All elements that have an edge that is not shared with other elements
        are removed.
        """
        
        frame_ids_to_delete = []
        for element in self.elements:
            if element.check_if_on_boundary():
                frame_ids_to_delete.append( element.id_in_frame )
                
        for frame_id in frame_ids_to_delete:
            self.delete_element_with_frame_id(frame_id)
    
    def delete_element_with_frame_id(self, frame_id):
        """Removes the element from the mesh entirely.
        
        Deletes the element from all shared nodes, remove all nodes belong to the element only.
        Deletes the element from element list, keeps internal frame id indexing updated.
        
        Parameters
        ----------
        
        frame_id : int
            id_in_frame of the element that is to be deleted
            
        See Also
        --------
        
        remove_element_with_frame_id : remove an alement without effecting the nodes
        """
        
        element_to_delete = self.get_element_with_frame_id( frame_id )
        
        nodes_to_remove = []

        for node in element_to_delete.nodes:
            if len(node.adjacent_elements) == 1:
                nodes_to_remove.append(node)
            else:
                node.remove_element_with_frame_id( frame_id )
        
        self.remove_list_of_nodes(nodes_to_remove)
        
        self.remove_element_with_frame_id( frame_id )
        
    def merge_short_edges(self, threshold_distance):
        """Merge short edges
        
        Merge short edges into single nodes. If multiple short edges are connected,
        merge them all into the same node.
        
        Parameters
        ----------
        
        threshold_distance : double
            the distance below which edges should be merged
        """

        # collect all edges that are shorter than the threshold distance
        edges = self.collect_edges()
        edges_to_merge = []
        for edge_index, edge in enumerate(edges):
            edge_length = np.linalg.norm( self.get_node_with_id( edge[0] ).position -
                                          self.get_node_with_id( edge[1] ).position )
            if edge_length < threshold_distance:
                edges_to_merge.append(edge)
        edges_np = np.array(edges_to_merge)
        
        # arrange short edges into clusters (pointslists)
        pointslists_to_merge = []
        edge_already_clustered = np.zeros(len(edges_to_merge),dtype = 'bool')

        for edge_index, edge in enumerate(edges_to_merge):
            if not edge_already_clustered[edge_index]:
                node_indices_to_merge = list(edge)
                still_extendable = True
                indices_to_keep = []
                while still_extendable:
                    still_extendable = False
                    for node_index in node_indices_to_merge:
                        occurences = np.where( edges_np == node_index )
                        edges_in_mersion = occurences[0]
                        new_node_indices_to_merge = np.unique(edges_np[edges_in_mersion])
                        for new_node_index in new_node_indices_to_merge:
                            if new_node_index not in node_indices_to_merge:
                                indices_to_keep.append(new_node_index)
                                still_extendable = True
                        edge_already_clustered[edges_in_mersion] = True
                    node_indices_to_merge += indices_to_keep
                pointslists_to_merge.append(node_indices_to_merge)

        # for each cluster, merge all nodes to their average position
        for pointslist in pointslists_to_merge:

            list_of_positions = []
            for point in pointslist:
                list_of_positions.append( self.get_node_with_id(point).position )
            new_node_position = np.mean(list_of_positions, axis = 0)

            # make new node
            maximal_node_id = self.get_maximal_node_id()
            new_node = Node(new_node_position, maximal_node_id + 1)
            self.nodes.append(new_node)

            # for each adjacent element of these nodes remove these nodes
            # from the element and add the new node instead
            for node_id in pointslist:
                this_node = self.get_node_with_id(node_id)
                for element in this_node.adjacent_elements:
                    new_node.adjacent_elements.append(element)
                    new_element_nodes = []
                    node_already_appended = False
                    for node in element.nodes:
                        if node.id != ( maximal_node_id + 1 ):
                            if node.id not in pointslist:
                                new_element_nodes.append(node)
                            else:
                                if not node_already_appended: 
                                    new_element_nodes.append(new_node)
                                    node_already_appended = True
                        else:
                            if not node_already_appended: 
                                new_element_nodes.append(new_node)
                                node_already_appended = True
                    element.nodes = new_element_nodes
            
            self.remove_list_of_nodes( [self.get_node_with_id(node_id) for node_id in pointslist] )
            
    def get_node_with_id(self, node_id):
        """Get the node instance that corresponds to this node id.
        
        Parameters
        ----------
        
        node_id : int
        
        Returns
        -------
        
        node_instance : node with id node_id
        """
        
        for node in self.nodes:
            if node.id == node_id:
                node_instance = node
                break
            
        return node_instance
    

class Element():
    """Elements are members of a mesh.
    """
    def __init__(self,nodes, id_in_frame = None):
        """The element generator.
            
        Parameters
        ----------
        nodes : list
            Each entry is a node    
        id_in_frame : int
            id of this element, defaults to None
            
        Returns
        -------
        element : the element
        """
        self.nodes = nodes
        """ A list containing all the nodes of this element"""
        
        self.id_in_frame = id_in_frame
        """ The id of this element within this particular frame"""
        
        self.global_id = None
        """ The id of this element globally"""
        
        # We add this element to all its nodes
        self.add_element_to_nodes()
        
    def add_element_to_nodes(self):
        """Adds the element to all its nodes, make sure to not do this twice!"""

        for node in self.nodes:
            node.adjacent_elements.append(self)

    def get_num_nodes(self):
        """Returns the number of nodes that this element shares.
        
        Returns
        -------
        
        number_of_shared_nodes : int
            number of nodes that are members of this element
        """

        return len(self.nodes)
    
    def check_if_on_boundary(self):
        """Return true if this element is on the boundary of the mesh.
        
        The element is on the boundary of the mesh if it has an edge that it
        does not share with another element. The check is done `in place' and
        this variable is not indexed.
        
        Returns
        -------
        
        element_is_on_boundary : bool
            True if element is on boundary, False if otherwise.
        """
        
        element_is_on_boundary = False
        
        if self.get_num_nodes() <= 2:
            element_is_on_boundary = True

        for local_index, this_node in enumerate(self.nodes):
            next_local_index = (local_index + 1)%self.get_num_nodes()
            next_node = self.nodes[next_local_index]
            frame_ids_this_node = this_node.get_adjacent_element_ids()
            frame_ids_next_node = next_node.get_adjacent_element_ids()
            shared_frame_ids = set.intersection( set(frame_ids_this_node), set(frame_ids_next_node) )
            if len(shared_frame_ids) == 1:
                element_is_on_boundary = True
                break
        
        return element_is_on_boundary


                
class Node():
    """Nodes are points in a mesh.
    """
    def __init__(self, position, id = None):
        """The node generator.
        
        Parameters
        ----------
        
        position : double array like
            position of the Node
        
        id : int
            id of the node, defaults to None
        
        Returns
        -------
        
        node : the node
        
        Warnings
        --------
        
        The equality method is hardcoded. That means, if you change or add node members you will need to 
        also implement that their equality is checked in the __eq__() method.
        """

        self.position = np.array(position, dtype = 'double')
        """The position of the node"""

        self.id = id
        """The id of the node"""

        self.adjacent_elements = []
        """A list with all the elements this node belongs to"""
        
    def get_adjacent_element_ids(self):
        """returns a list of ids of all containing elements
        
        Returns
        -------
        
        id_list : int list
            list containing ids of all adjacent elements
        """
        
        id_list = []

        for element in self.adjacent_elements:
            id_list.append(element.id_in_frame)
        
        return id_list

    def remove_element_with_frame_id(self, id_in_frame ):
        """Remove element with id_in_frame from list of adjacent elements.
        
        Parameters
        ----------
        
        id_in_frame : int
            id of element that is to be removed from the list of adjacent elements
            for this node
        """
        self.adjacent_elements = [element for element in self.adjacent_elements if 
                                  element.id_in_frame != id_in_frame]
        

def get_contour_list(this_image):
    """Get a list of contours around each cell in the image.
    
    Uses the opencv function cv2.findContours
    
    Parameters
    ----------
    
    this_image : ndarray
        an image as integer numpy array

    Returns
    -------
    
    contour_list : list
        each entry is a contour as returned by the opencv function
        
    cell_ids : list
        entries are the integer values of segmented objects in the data frame.
        order of this list is the same as in contour_list
    """

    cell_ids = np.unique( this_image )[1:] # Skip background

    # collect contours, i.e. for each region we get the coordinates of the outer-most set of
    # pixels
    contour_list = [None]*len(cell_ids)

    for cell_number, cell_id in enumerate(cell_ids):
        # Get an image with ones for the current cell and zeros outside
        cell_mask = ( this_image == cell_id )
        # Transform to 8 bit for openCV
        cell_mask = np.uint8( cell_mask )
        contour_list[cell_number],_ = cv2.findContours( cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )    

    return contour_list, cell_ids
    
def find_triple_junctions_at_pixel(this_image, x_index, y_index):
    """Finds triple junctions in a segmented image around the pixel with coordinates (x_index, y_index)
    
    In this function, a triple junction is indexed by the pixel to the lower left pixel corner of where three
    pixels of different colour meet. Coordinates in the input and output of this function are like this:
    
            _____y______>
           |
          x|   IMAGECONTENT
           |
           \/
           
    Coordinates are different in all other functions in this package: in extract_vertex_data we 
    change to cartesian coordinates.
    
    See Also
    --------
        extract_vertex_data

    Parameters
    ----------
    
    this_image : ndarray
        an image as integer numpy array
        
    x_index : int
        x_coordinate of this pixel
        
    y_index : int 
        y_coordinate of this pixel

    Returns
    -------
    
    odered_triple_junctions: list of [x,y] coordinates
        ordered list of triple junctions, ordered anticlockwise starting
        at the first corner of the 'first' anticlockwise edge. This is the
        anticlockwise edge after an edge that is shared with a pixel of the 
        same colour.
    """

    values_in_neighbourhood = this_image[(x_index -1):(x_index + 2)][:,(y_index - 1):(y_index + 2)]
    central_value = values_in_neighbourhood[1,1]
    triple_junctions = []

    if ( central_value != values_in_neighbourhood[0,0] and central_value != values_in_neighbourhood[0,1] and
         values_in_neighbourhood[0,0] != values_in_neighbourhood [0,1] ): # top left

        triple_junctions.append( np.array([x_index, y_index -1]) )

    if ( central_value != values_in_neighbourhood[0,0] and central_value != values_in_neighbourhood[1,0] and
         values_in_neighbourhood[0,0] != values_in_neighbourhood [1,0] ): # top left

        triple_junctions.append( np.array([x_index, y_index -1]) )

    if ( central_value != values_in_neighbourhood[1,0] and central_value != values_in_neighbourhood[2,0] and
         values_in_neighbourhood[1,0] != values_in_neighbourhood [2,0] ): # bottom left

        triple_junctions.append( np.array([x_index +1, y_index -1]) )

    if ( central_value != values_in_neighbourhood[2,0] and central_value != values_in_neighbourhood[2,1] and
         values_in_neighbourhood[2,0] != values_in_neighbourhood [2,1] ): # bottom left

        triple_junctions.append( np.array([x_index +1, y_index -1]) )

    if ( central_value != values_in_neighbourhood[2,1] and central_value != values_in_neighbourhood[2,2] and
         values_in_neighbourhood[2,1] != values_in_neighbourhood [2,2] ): # bottom right

        triple_junctions.append( np.array([x_index +1, y_index]) )

    if ( central_value != values_in_neighbourhood[2,2] and central_value != values_in_neighbourhood[1,2] and
         values_in_neighbourhood[2,2] != values_in_neighbourhood [1,2] ): # bottom right

        triple_junctions.append( np.array([x_index +1, y_index]) )

    if ( central_value != values_in_neighbourhood[0,1] and central_value != values_in_neighbourhood[0,2] and
         values_in_neighbourhood[0,1] != values_in_neighbourhood [0,2]): # top right

        triple_junctions.append( np.array([x_index, y_index]) )

    if ( central_value != values_in_neighbourhood[0,2] and central_value != values_in_neighbourhood[1,2] and
         values_in_neighbourhood[0,2] != values_in_neighbourhood [1,2]): # top right

        triple_junctions.append( np.array([x_index, y_index]) )

    ### ORDERING corners here
        
    ordered_triple_junctions = []
    triple_junctions = np.array(triple_junctions)

    if ( len( triple_junctions ) > 1 ):
        
        if ( central_value == values_in_neighbourhood[0,1] and central_value != values_in_neighbourhood[1,0] ):

            if np.any(np.all( triple_junctions == [x_index, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index -1] )
            if np.any(np.all( triple_junctions == [x_index +1, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index -1])
            if np.any(np.all( triple_junctions == [x_index +1, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index])

        if central_value == values_in_neighbourhood[1,0] and central_value != values_in_neighbourhood[2,1]:

            if np.any(np.all( triple_junctions == [x_index +1, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index -1])
            if np.any(np.all( triple_junctions == [x_index +1, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index -1] )

        if central_value == values_in_neighbourhood[2,1] and central_value != values_in_neighbourhood[1,2]:

            if np.any(np.all( triple_junctions == [x_index +1, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index -1] )
            if np.any(np.all( triple_junctions == [x_index +1, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index -1])

        if central_value == values_in_neighbourhood[1,2] and central_value != values_in_neighbourhood[0,1]:

            if np.any(np.all( triple_junctions == [x_index, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index])
            if np.any(np.all( triple_junctions == [x_index, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index, y_index -1] )
            if np.any(np.all( triple_junctions == [x_index +1, y_index -1] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index -1])
            if np.any(np.all( triple_junctions == [x_index +1, y_index] , axis = 1)):
                ordered_triple_junctions.append( [x_index +1, y_index])
    else:
        ordered_triple_junctions = triple_junctions
        
    return ordered_triple_junctions

def tiff2vertex(filename):
    # Open the presegmented tiff and get the contours of the cells
    this_image = plt.imread(filename)
    contour_list, cell_ids = get_contour_list(this_image)

    # Get number of cells
    no_of_cells = len(cell_ids)

    # We overestimate the size of this array by approximately a factor of 2
    number_of_vertices_estimate = no_of_cells*7/2
    vertex_array = np.zeros( ( number_of_vertices_estimate, 2 ), dtype=np.int64 )
    no_of_vertices = 0
    no_of_cells_per_vertex = np.zeros(number_of_vertices_estimate, dtype=np.int64)

    # We also overestimate the maximal number of vertices that a cell can have
    vertices_of_cells = np.zeros( (no_of_cells,30), dtype=np.int64)
    number_vertices_of_cells = np.zeros(no_of_cells, dtype=np.int64)

    # We loop over each cell
    for contour_index in range(no_of_cells):
        no_of_vertices_this_cell = 0
        # We loop over each border pixel of that cell
        for pixel_index in range(np.shape(contour_list[contour_index])[1]):

            y_index = contour_list[contour_index][0][pixel_index][0][0]
            x_index = contour_list[contour_index][0][pixel_index][0][1] # why are the x and y coords flipped? Work around later in function...

            # Get all the surrounding pixels of the current pixel
            values_in_neighbourhood = this_image[(x_index -1):(x_index + 2)][:,(y_index - 1):(y_index + 2)]
            no_values_in_neighbourhood = len(np.unique(values_in_neighbourhood))

            if (no_values_in_neighbourhood > 2):
                ordered_triple_junctions = find_triple_junctions_at_pixel(this_image, x_index, y_index)
                for triple_junction in ordered_triple_junctions:
                    if np.any( np.all( vertex_array == triple_junction, axis =1 ) ):
                        #This_junction has been found before, here is the index of that vertex
                        vertex_index = np.where( np.all( vertex_array == triple_junction, axis = 1 ))[0][0]
                        # The Triple Junction is potentially in the Neighbourhood of multiple pixels of a cell
                        # Check whether it is member of the cell already
                        if no_of_vertices_this_cell > 0:
                            if ( vertex_index not in 
                                 vertices_of_cells[contour_index,0:number_vertices_of_cells[contour_index]]):
                                # Ok, it was not in the neighbourhood of the last pixel, so we can
                                # assign the Vertex to this cell
                                no_of_cells_per_vertex[ vertex_index ]  += 1
                                vertices_of_cells[contour_index, no_of_vertices_this_cell] = vertex_index                                
                                number_vertices_of_cells[contour_index] += 1
                                no_of_vertices_this_cell += 1
                        else:
                            # This is the first vertex for this cell, so let's assign the vertex to
                            # this cell
                            no_of_cells_per_vertex[ vertex_index ] += 1
                            vertices_of_cells[contour_index, no_of_vertices_this_cell] = vertex_index
                            number_vertices_of_cells[contour_index] += 1
                            no_of_vertices_this_cell += 1
                    else:
                        # This is a new vertex. Let's write that vertex and assign it to this cell
                        vertex_array[ no_of_vertices ] = triple_junction
                        no_of_cells_per_vertex[no_of_vertices] +=1

                        vertices_of_cells[ contour_index, no_of_vertices_this_cell] = no_of_vertices
                        no_of_vertices_this_cell += 1
                        number_vertices_of_cells[contour_index] += 1

                        no_of_vertices +=1
    
    # We now have some vectors that are too long. Let's shorten them!
    # We also need move the vertex positions from the pixel to the actual
    # junction
    
    no_of_cells_per_vertex = no_of_cells_per_vertex[0:no_of_vertices]
    vertices_of_cells = vertices_of_cells[0:no_of_vertices]
    
    # adjust coordinates of actual vertex positions
    vertex_array = np.float64(vertex_array[0:no_of_vertices])
    vertex_array = vertex_array + [-0.5,0.5]

    # move to Cartesian coordinates
    new_vertex_array = np.zeros_like(vertex_array)
#    new_vertex_array[:,0] = vertex_array[:,1]
#    new_vertex_array[:,1] = len(this_image[:,0]) - vertex_array[:,0]
    new_vertex_array[:,0] = vertex_array[:,1] # This gives the correct x, y coordinates - no idea what the previous two lines are doing.
    new_vertex_array[:,1] = vertex_array[:,0]

    # Build mesh
    nodes = []
    for vertex_position in new_vertex_array:
        nodes.append( Node( vertex_position ) )
        
    elements = []
    for cell_index, row in enumerate( vertices_of_cells ):
        element_nodes = []
        for local_index in range( 0, number_vertices_of_cells[cell_index] ):
            element_nodes.append( nodes[row[local_index]] )
        elements.append( Element( element_nodes, cell_ids[ cell_index ] ) )
       
    this_mesh = Mesh( nodes, elements )
    this_mesh.merge_short_edges(2)
    this_mesh.remove_boundary_elements()
    this_mesh.assign_node_ids_in_order()

    return this_mesh.nodes, this_mesh.elements


def tiff2immersed(filename, margin=5, scaling=1.0):
    # Open the presegmented tiff and get the contours of the cells
    this_image = plt.imread(filename)
    contour_list, cell_ids = get_contour_list(this_image)

    # Get dimensions of segmented image
    height, width = this_image.shape

    new_vertex_array = []
    vertices_of_cells = []
    number_vertices_of_cells = []

    current_index = 0
    for contour in contour_list:
        # If any pixel lies within 'margin' of the image boundary, cull this element
        on_boundary = False
        for pixel_index in range(np.shape(contour)[1]):
            pixel_x = contour[0][pixel_index][0][0]
            pixel_y = contour[0][pixel_index][0][1]
            if pixel_x <= margin:
                on_boundary = True
                break
            if pixel_y <= margin:
                on_boundary = True
                break
            if pixel_x >= width - margin:
		on_boundary = True
                break
            if pixel_y >= height - margin:
                on_boundary = True
                break 
        if on_boundary == True:
            continue

	# Calculate centroid of this contour
	centroid_x = 0
	centroid_y = 0
	num_indices = 0
        for pixel_index in range(np.shape(contour)[1]):
            pixel_x = contour[0][pixel_index][0][0]
            pixel_y = contour[0][pixel_index][0][1]
	    centroid_x += pixel_x
	    centroid_y += pixel_y
	    num_indices += 1
	centroid_x /= num_indices
	centroid_y /= num_indices

	# Collect every pixel as a node for this element
        indices = []
        for pixel_index in range(np.shape(contour)[1]):
            pixel_x = contour[0][pixel_index][0][0]
            pixel_y = contour[0][pixel_index][0][1]

	    # Rescale nodes relative to centroid to produce a space margin between elements
	    rescaled_x = centroid_x + (pixel_x - centroid_x) * scaling
	    rescaled_y = centroid_y + (pixel_y - centroid_y) * scaling

            new_vertex_array.append([rescaled_x, rescaled_y])
            indices.append(current_index)
            current_index += 1
        vertices_of_cells.append(indices)
        number_vertices_of_cells.append(len(indices))

        # Create nodes array
	nodes = []
	for vertex_position in new_vertex_array:
		nodes.append(Node(vertex_position))

        # Create elements array
	elements = []
	cell_index = 0
	for cell in vertices_of_cells:
		element_nodes = []
		for node_index in cell:
			element_nodes.append(nodes[node_index])
		elements.append(Element(element_nodes, cell_ids[cell_index]))
		cell_index += 1

        # Make mesh
        this_mesh = Mesh( nodes, elements )
	this_mesh.assign_node_ids_in_order()

    return this_mesh.nodes, this_mesh.elements


def tiff2spheres(filename, margin=5):
    # Open the presegmented tiff and get the contours of the cells
    this_image = plt.imread(filename)
    contour_list, cell_ids = get_contour_list(this_image)

    # Get dimensions of image
    height, width = this_image.shape

    new_vertex_array = []
    vertices_of_cells = []
    number_vertices_of_cells = []

    current_index = 0
    for contour in contour_list:
	# If any pixel lies within 'margin' of the image boundary, cull this element
        on_boundary = False
        for pixel_index in range(np.shape(contour)[1]):
            pixel_x = contour[0][pixel_index][0][0]
            pixel_y = contour[0][pixel_index][0][1]
            if pixel_x <= margin:
                on_boundary = True
                break
            if pixel_y <= margin:
                on_boundary = True
                break
            if pixel_x >= width - margin:
		on_boundary = True
                break
            if pixel_y >= height - margin:
                on_boundary = True
                break 
        if on_boundary == True:
            continue

        # Calculate centroid of this contour
        centroid_x = 0
        centroid_y = 0
        num_indices = 0
        for pixel_index in range(np.shape(contour)[1]):
            pixel_x = contour[0][pixel_index][0][0]
            pixel_y = contour[0][pixel_index][0][1]
	    centroid_x += pixel_x
	    centroid_y += pixel_y
	    num_indices += 1
	centroid_x /= num_indices
	centroid_y /= num_indices
	new_vertex_array.append([centroid_x, centroid_y])

    # Create nodes array
    nodes = []
    for vertex_position in new_vertex_array:
        nodes.append(Node(vertex_position))

    elements = []

    # Make mesh
    this_mesh = Mesh( nodes, elements )
    this_mesh.assign_node_ids_in_order()

    return this_mesh.nodes

def tiff2potts(filename, margin=5):
        # Open the presegmented tiff and get the contours of the cells
        this_image = plt.imread(filename)
        contour_list, cell_ids = get_contour_list(this_image)

	# Do an initial pass to determine which colours to ignore
	height, width = this_image.shape
	ignore_colours = []
	for y in range(height):
		for x in range(width):
			colour = this_image[y][x]
			if x < margin or x > width - margin or y < margin or y > height - margin:
				if colour not in ignore_colours:
					ignore_colours.append(colour)

	# Build up the elements dictionary with any (non-ignored) colours
	vertices = []
	elementsDict = {}
	node_index = 0
	for y in range(height):
		for x in range(width):
			vertices.append([x,y])
			colour = this_image[y][x]

			# Check if node is a free node (not part of an element)
			if colour in ignore_colours:
				node_index += 1
				continue

			if colour not in elementsDict.keys():
				elementsDict[colour] = [node_index]
			else:
				elementsDict[colour].append(node_index)
			node_index += 1

        # Create nodes array
	nodes = []
	for vertex_position in vertices:
		nodes.append(Node(vertex_position))

        # Create elements array
	elements = []
	cell_index = 0
	for colour in elementsDict.keys():
		element_nodes = []
		for node_index in elementsDict[colour]:
			element_nodes.append(nodes[node_index])
		elements.append(Element(element_nodes, colour))
		cell_index += 1

        # Make mesh
        this_mesh = Mesh(nodes, elements)
	this_mesh.assign_node_ids_in_order()

	return this_mesh.nodes, this_mesh.elements

""" Write CHASTE vertex simulation input files for the given node and element list."""
def write_vertex_files(nodes, elements, outnode, outcell):
	print "Writing Vertex Simulation files:"
	num_nodes = len(nodes)
	num_elements = len(elements)
	num_dimensions = 2
	num_node_attributes = 0
	num_element_attributes = 0
	max_boundary_marker = 0

	print "Writing node file..."
	with open(outnode, "w") as outfile:
		node_header = [str(num_nodes), str(num_dimensions), str(num_node_attributes), str(max_boundary_marker)]
		outfile.write('\t'.join(node_header) + '\n')
		for index, node in enumerate(nodes):
			x = node.position[0]
			y = node.position[1]
			outfile.write('\t'.join([str(index), str(x), str(y)]) + '\n')
	print "...done."

	print "Writing cell file..."
	element_header = [str(num_elements), str(num_element_attributes)]
	with open(outcell, "w") as outfile:
		outfile.write('\t'.join(element_header) + '\n')
		for index, element in enumerate(elements):
			num_element_nodes = len(element.nodes)
			indices = []
			for node in element.nodes:
				indices.append(str(node.id))
			indices = '\t'.join(indices)
			outfile.write('\t'.join([str(index), str(num_element_nodes), indices, str(0)]) + '\n')
	print "...done."

""" Write CHASTE immersed boundary simulation input files for the given node and element list."""
def write_immersed_files(nodes, elements, outnode, outcell):
	print "Writing Immersed Boundary Simulation files:"
	num_nodes = len(nodes)
	num_elements = len(elements)
	num_dimensions = 2
	num_node_attributes = 0
	num_element_attributes = 0
	max_boundary_marker = 0

	print "Writing node file..."
	with open(outnode, "w") as outfile:
		node_header = [str(num_nodes), str(num_dimensions), str(num_node_attributes), str(max_boundary_marker)]
		outfile.write('\t'.join(node_header) + '\n')
		for index, node in enumerate(nodes):
			x = node.position[0]
			y = node.position[1]
			outfile.write('\t'.join([str(index), str(x), str(y)]) + '\n')
	print "...done."

	print "Writing cell file..."
	element_header = [str(num_elements), str(num_element_attributes)]
	with open(outcell, "w") as outfile:
		outfile.write('\t'.join(element_header) + '\n')
		for index, element in enumerate(elements):
			num_element_nodes = len(element.nodes)
			indices = []
			for node in element.nodes:
				indices.append(str(node.id))
			indices = '\t'.join(indices)
			outfile.write('\t'.join([str(index), str(num_element_nodes), indices, str(0)]) + '\n')
	print "...done."


""" Write CHASTE Potts simulation input files for the given node and element list."""
def write_potts_files(nodes, elements, outnode, outcell):
	print "Writing Potts Simulation files:"
	num_nodes = len(nodes)
	num_elements = len(elements)
	num_dimensions = 2
	num_node_attributes = 0
	num_element_attributes = 0
	max_boundary_marker = 1

	print "Writing node file..."
	with open(outnode, "w") as outfile:
		node_header = [str(num_nodes), str(num_dimensions), str(num_node_attributes), str(max_boundary_marker)]
		outfile.write('\t'.join(node_header) + '\n')
		for index, node in enumerate(nodes):
			x = node.position[0]
			y = node.position[1]
			outfile.write('\t'.join([str(index), str(x), str(y), str(0)]) + '\n')
	print "...done."

	print "Writing cell file..."
	element_header = [str(num_elements), str(num_element_attributes)]
	with open(outcell, "w") as outfile:
		outfile.write('\t'.join(element_header) + '\n')
		for index, element in enumerate(elements):
			num_element_nodes = len(element.nodes)
			indices = []
			for node in element.nodes:
				indices.append(str(node.id))
			indices = '\t'.join(indices)
			outfile.write('\t'.join([str(index), str(num_element_nodes), indices, str(0)]) + '\n')
	print "...done."

""" Write the CHASTE overlapping spheres simulation input file (nodes only) for the given node list."""
def write_spheres_files(nodes, outnode):
	print "Writing Overlapping Spheres Simulation files:"
	num_nodes = len(nodes)
	num_dimensions = 2
	num_node_attributes = 0
	max_boundary_marker = 1

	print "Writing node file..."
	with open(outnode, "w") as outfile:
		node_header = [str(num_nodes), str(num_dimensions), str(num_node_attributes), str(max_boundary_marker)]
		outfile.write('\t'.join(node_header) + '\n')
		for index, node in enumerate(nodes):
			x = node.position[0]
			y = node.position[1]
			outfile.write('\t'.join([str(index), str(x), str(y), str(0)]) + '\n')
	print "...done."
