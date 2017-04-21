import numpy as np
from numpy import linalg as la
from scipy.spatial import Delaunay, distance
from scipy.stats import norm as normal
from PIL import Image, ImageDraw
import cv2
import itertools
import argparse
import random
import copy
import json

def bilateral_filter( img, diameter, scolor, sspace, minthresh, ratio ):
    """
    Blurs the borders, in order to remove lines drawn in the image, or
    artifacts because the fabrication method.
    """
    bilateral = cv2.bilateralFilter(img, diameter, scolor, sspace)
    return bilateral

def calculate_edge_points( img, minthresh, ratio ):
    """
    Calculates the border from the image. It uses OpenCV Canny filter.
    Default parameters seem to be good for all training cases.
    Parameters
    ----------
    img   : np.array
        The RGB image file.
    minthresh : int
        lower threshold of the Canny filter. Recommended 50
    ratio : int
        ratio to the upper threshold of the Canny filter. Canny recommended from 2 to 3.
    Returns
    -------
        The detected points part of the edges.
    """
    # Get the grayscale image.
    points = set()
    edges = cv2.Canny(img, minthresh, minthresh*ratio)
    for x in xrange(edges.shape[0]):
        for y in xrange(edges.shape[1]):
            if edges[x,y] == 255:
                points.add((y,x))
    
    return list(points)

def relative_neighborhood_graph( points ):
    """
    Calculates the relative neighborhood graph from a set of points.
    There are algorithms that might do a better job at this, but 
    for the purposes of sketching a solution, this is good enough.
    Parameters
    ----------
    points : list
        the (x, y) coordinates of the points to find the relative neighborhood graph.
    Returns
    -------
    dict
        The dictionary with edges as keys and edge distances as value. The edge distances
        are necessary in every following step, so they are saved.
    """
    # Calculate the Delaunay triangulation.
    triangulation = Delaunay(points)
    
    # Get all its edges
    edgs = set()
    cons = {}
    for tri in triangulation.simplices:
        triedgs = []
        for edg in itertools.combinations(tri,2):
            edg = tuple(sorted(edg))
            # Calculate the connectivity of the DT.
            
            for i in range(2):
                if not edg[i] in cons:
                    cons[edg[i]] = []
                if not edg[(i+1)%2] in cons[edg[i]]:
                    cons[edg[i]].append(edg[(i+1)%2])
            edgs.add(edg)
    
    rng = {}
    # Calculate the nearest neighbor graph.
    for edg in edgs:
        d = distance.euclidean(points[edg[0]], points[edg[1]])
        ns = set()
        for n in range(2):
            for c in cons[edg[n]]:
                if c != edg[(n+1)%2]:
                    ns.add(c)
        for n in ns:
            if distance.euclidean(points[edg[0]], points[n]) < d and \
                distance.euclidean(points[edg[1]], points[n]) < d:
                break
        else:
            rng[edg] = d
    return rng

def filter_by_normal_distribution(edgs, cut):
    """
    It filters edges by a distance, calculated using the normal distribution.
    Parameters
    ----------
    edgs : dict
        The edges previously filtered and with their lenght added to them.
    cut  : real
        The cut to discard in percentage
    Returns
    -------
    dict
        The output dict filtered by the distance calculated using the normal distribution.
    """
    # Calculate the mean and standard deviation.
    ds = np.array(edgs.values())
    np.sort(ds)
    mean = np.mean(ds)
    std = np.std(ds)
    # Calculate the cut distance.
    cutd = mean + normal.ppf(cut/100.0) * std
    # Filter the values by the distance.
    out = { k: v for k, v in edgs.iteritems() if v < cutd }
    return out

def filter_by_full_point(edgs, cut):
    """
    It filters edge e0 if one of its points has two edges, e1, e2
    and the distance is above the sum of those two edges multiplied
    by cut. (edgs[e0] > (edges[e1]+edges[e2])*cut
    Parameters
    ----------
    edgs : dict:
        A set of edges to filter.
    cut : the number to multiply the neighbor edges with.
    Results
    -------
    dict : the dictionary after filtering the bad edges.
    """
    pnh = {}
    
    for edg, d in edgs.iteritems():
        for p in edg:
            if not p in pnh:
                pnh[p] = [(edg, d)]
            else:
                pnh[p].append((edg, d))
    
    bedgs = set()
    
    for p, es in pnh.iteritems():
        if len(es) > 2:
            ses = sorted(es, key=lambda e: e[1])
            mn = ses[0][1]+ses[1][1]
            for e in ses[2:]:
                if e[1] > mn*cut:
                    bedgs.add(e[0])
                if len(pnh[e[0][0]]) > 2 and len(pnh[e[0][1]]) > 2:
                    bedgs.add(e[0])
    
    return { k: v for k, v in edgs.iteritems() if not k in bedgs }

def separate_lines(edgs):
    """
    It creates lines from the edges. Whenever it finds a line end,
    or a line that splits in more than two, it stops.
    Parameters
    ----------
    edgs : dict
        A set of edges to join together.
    Results
    -------
    dict : the edges joined at the corners.
    """
    # Find the points with respective edges.
    pnh = {}
    for edg, d in edgs.iteritems():
        for p in edg:
            if not p in pnh:
                pnh[p] = [(edg, d)]
            else:
                pnh[p].append((edg, d))
    
    rem_edges = set(edgs.keys())
    def new_line_edge():
        """
        Picks the next best edge to start a new line.
        """
        e3 = None
        e2 = None
        for edg in rem_edges:
            n0 = edg[0]
            n1 = edg[1]
            c0 = len(pnh[n0])
            c1 = len(pnh[n1])
            # Return immediately if it's the end of a line.
            if c0 == 1:
                return (n0, n1)
            elif c1 == 1:
                return (n1, n0)
            # Save to return if it's in a 3 point.
            if e3 is None and c0 > 2:
                e3 = (n0, n1)
            if e3 is None and c1 > 2:
                e3 = (n1, n0)
            # Save to return if it's in a 2 point.
            if e2 is None and c0 == 2 and c1 == 2:
                e2 = (n0, n1)
        
        if e3 is not None:
            return e3
        return e2
    
    
    def remove_edge(e):
        """
        Removes an edge from the set of edges, even if it's in the oposite direction.
        """
        if e in rem_edges:
            rem_edges.remove(e)
        else:
            if (e[1], e[0]) in rem_edges:
                rem_edges.remove((e[1], e[0]))
            else:
                raise Exception("not in removed") 
    equal_edge = lambda e1, e2: (e1[0] == e2[0] and e1[1] == e2[1]) or \
                                (e1[1] == e2[0] and e1[0] == e2[1])
    
    def next_edge(e):
        """
        Picks the next edge from this node or None if it's in 
        the end of a line or if the corner is divided in 3.
        """
        # Get possible edges from the list.
        pose = pnh[e[1]]
        
        # If the possible edges are more than 2 or less than two, return None.
        if len(pose) != 2:
            return None
        # Now, for all the possible e, 
        for pe, d in pose:
            if equal_edge(pe, e) or not pe in rem_edges:
                continue
            if pe[1] == e[1]:
                return (pe[1], pe[0])
            return (pe[0], pe[1])
        return None
    
    # Goes through every edge, getting the next.
    e = new_line_edge()
    lines = []
    while ( e is not None ):
        lines.append(list(e))
        remove_edge(e)
        ne = next_edge(e)
        while( ne is not None ):
            lines[-1].append(ne[1])
            remove_edge(ne)
            ne = next_edge(ne)
        e = new_line_edge()
    
    return lines

def classify_lines(lines, points, window, ratio, maxfit):
    """
    Checks all the lines and classifies them as straight or not.
    Parameters
    ----------
    lines : list
        The list of detected lines.
    points : list
        The list of points.
    window : int
        The window to fit the least squares.
    ratio : float
        The ratio between a curve and a straight line to know it fitted.
    maxfit : float
        The maximum avg sq to declare this line unclassified.
    """
    
    def fit3(x, y, p):
        ye = p[0]*np.power(x,3) + p[1]*np.power(x,2) + p[2]*x + p[3]
        s = np.sum(np.power(ye-y,2))
        return s/x.shape[0]
    
    def fit1(x, y, p):
        ye = x*p[0] + p[1]
        s = np.sum(np.power(ye-y,2))
        return s/x.shape[0]
    
    to_class = copy.deepcopy(lines)
    classi = {'other': [],
              'straight': [],
              'curve': [] }
    
    def get_line_curve_fit(subl):
        p0 = np.array(points[subl[0]], dtype=float)
        p1 = np.array(points[subl[-1]], dtype=float)
        v = p1-p0
        nor = la.norm(v)
        if nor < 1e-3:
            return None
        v /= nor
        vp = np.array([v[1], -v[0]])
        st = map( lambda n: np.array(points[n], dtype=float), subl )
        x = np.array(map( lambda p: np.dot(p-p0, v), st))
        y = np.array(map( lambda p: np.dot(p-p0, vp), st))
        f1 = np.polyfit(x, y, 1)
        f3 = np.polyfit(x,y,3)
        s1 = fit1(x,y,f1)
        s3 = fit3(x,y,f3)
        return ((f1,s1), (f3,s3), (v, vp, p0))
    
    def is_lin_fit(pt, linfit):
        c1, c0, v, vp, p0 = linfit
        p = np.array(points[line[i]])-p0
        x = np.dot(p, v)
        y = np.dot(p, vp)
        ye = c1*x + c0
        if (y-ye)**2 < ratio*mres:
            return True
        else:
            return False
            
    while ( len(to_class) ):
        line = to_class.pop(0)
        mres = float('inf')
        linfit = None
        mi = None
        mres3 = float('inf')
        mi3 = None
        for i in xrange(0, len(line), window):
            if i + window > len(line) and mi is not None:
                continue
            subl = line[i:i+window]
            res = get_line_curve_fit(subl)
            if res is None:
                continue
            ((f1, s1), (f3, s3), (v, vp, p0)) = res
            if s1 < mres:
                mres = s1
                mi = i
                linfit = [f1[0], f1[1], v, vp, p0]
            if s3 < mres3:
                mres3 = s3
                mi3 = i
        if mres > maxfit and mres3 > maxfit:
            classi['other'].append(line)
        elif mres3*ratio < mres:
            # In the case of a straight line, I don't have to worry about the line changing fast and not fitting,
            # So I just test the new points against the previous linear fit.
                
            beg = mi
            end = min(mi+window, len(line))
            for i in xrange(beg-1, -1, -1):
                if is_lin_fit(points[line[i]], linfit):
                    beg -= 1
                else:
                    break
            for i in xrange(end, len(line)):
                if is_lin_fit(points[line[i]], linfit):
                    end += 1
                else:
                    break
            if beg > 0:
                to_class.append(line[:beg])
            if end < len(line):
                to_class.append(line[end:])
            classi['straight'].append(line[beg:end])
        else:
            # In the case of a curve, it might turn completely, or do something weird, so it's better to try to fit moving
            # with the back window.
            beg = mi
            end = min(mi+window, len(line))
            cline = line[beg:end]
            for i in xrange(beg-int(window/3), -1, -int(window/3)):
                mnl = max(i, 0)
                subl = line[mnl:mnl+window]
                res = get_line_curve_fit(subl)
                if res is None:
                    break
                ((f1, s1), (f3,s3), (v, vp, p0)) = res
                if s1 < ratio*mres3:
                    beg = mnl
                else:
                    break
            for i in xrange(end+int(window/3), len(line), int(window/3)):
                mxl = min(i+window, len(line))
                subl = line[i:mxl]
                res = get_line_curve_fit(subl)
                if res is None:
                    break
                ((f1, s1), (f3,s3), (v, vp, p0)) = res
                if s1 < ratio*mres3:
                    end = mxl
                else:
                    break
            if beg > 0:
                to_class.append(line[:beg])
            if end < len(line):
                to_class.append(line[end:])
            classi['curve'].append(line[beg:end])
    return classi

def painted_image( img, edgs={}, points={} ):
    """
    Paints an image with the given edges.
    Parameters
    ----------
    img  : np.array 
        Array containing the image to paint over.
    edgs : dict
        dictionary containing every edge in the keys and its color in the values.
    Returns
    -------
        PIL Image
        The image with the drawn lines with the given color.
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for edg, c in edgs.iteritems():
        draw.line(edg, fill=c)
    
    for point, c in points.iteritems():
        draw.ellipse((point[0], point[1], point[0]+1, point[1]+1), outline=c)
    
    return image

def random_color():
    return ( random.randrange(256), random.randrange(256), random.randrange(256))

def calculate_lines_and_curves(img, preprocess, canny, normal, full, line, debug):
    """
    Calculates the straight lines and curves of an image. 
    Parameters
    ----------
    imgname : open cv image.
        The image to process.
    canny : dict
        The parameters for the canny edge detection algorithm.
    normal : dict
        The parameters for the normal edge filter or None
    full : dict
        The parameters for the full point edge filter or None
    line : dict
        The parameters for the straight line detection.
    curve : dict
        The parameters for the curve detection.
    Results
    -------
    dict { 'curves': [[(x11,y11),(x12,y12),...], [(x21, y21),...], ...],
           'lines':  [[(x11,y11),(x12,y12),...], [(x21, y21),...], ...] }
    """

    if preprocess is not None:
        args = preprocess.copy()
        args.update(canny)
        img = bilateral_filter(img, **args)
        if debug:
            draw = painted_image( img )
            draw.show(title="Preprocessed Image")
            raw_input("Press key...")

    points = calculate_edge_points(img, **canny)
    if debug:
        draw = painted_image( img, points={ k: (255,0,0) for k in points } )
        draw.show(title="Image with Canny points")
        raw_input("Press key...")

    edgs = relative_neighborhood_graph(points)
    if debug:
        draw = painted_image( img, edgs={ (points[k[0]], points[k[1]]): (255,0,0) for k in edgs } )
        draw.show(title="Image with Relative Neighborhood Graph")
        raw_input("Press key...")
    
    if normal is not None:
        edgs = filter_by_normal_distribution(edgs, **normal)
        if debug:
            draw = painted_image( img, edgs={ (points[k[0]], points[k[1]]): (255,0,0) for k in edgs } )
            draw.show(title="Image after normal filter.")
            raw_input("Press key...")
    
    if full is not None:
        edgs = filter_by_full_point(edgs, **full)
        if debug:
            draw = painted_image( img, edgs={ (points[k[0]], points[k[1]]): (255,0,0) for k in edgs } )
            draw.show(title="Image after full edges filter.")
            raw_input("Press key...")
    
    lines = separate_lines(edgs)
    if debug:
        draw = painted_image( img, edgs={ tuple(map(lambda n: points[n], l)): random_color() for l in lines } )
        draw.show(title="Image: after separating edges.")
        raw_input("Press key...")
    lines = classify_lines(lines, points, **line) 
    
    dedgs = {}
    lineso = [map(lambda n: points[n], l) for l in lines['other']]
    liness = [map(lambda n: points[n], l) for l in lines['straight']]
    linesc = [map(lambda n: points[n], l) for l in lines['curve']]
    edgso = { tuple(l): (0,255,0) for l in lineso }
    edgss = { tuple(l): (255,0,0) for l in liness }
    edgsc = { tuple(l): (0,0,255) for l in linesc }
    dedgs.update(edgso)
    dedgs.update(edgss)
    dedgs.update(edgsc)
    draw = painted_image( img, edgs=dedgs )
    if debug:
        draw.show(title="Image: after detecting lines, curves, other.")
        raw_input("Press key...")
    return ( {'other': lineso, 'straight': liness, 'curve': linesc}, draw )
    
def execute_task( args ):
    name = args.input.name
    outname = args.output.name
    args.input.close()
    args.output.close()
    img = cv2.imread(name)
    
    if args.no_bilateral:
        preprocess = None
    else:
        preprocess = { 'diameter': args.bilateral_diameter, 
                       'sspace': args.bilateral_sspace, 
                       'scolor': args.bilateral_scolor }
    
    canny = { 'minthresh': args.canny_minthresh, 'ratio': args.canny_ratio }
    
    if args.no_normal:
        normal = None
    else:
        normal = { 'cut': args.normal_cut }
    
    if args.no_full:
        full = None
    else:
        full = { 'cut': args.full_cut }
    
    line = { 'window': args.line_window, 'ratio': args.line_ratio, 'maxfit': args.line_maxfit }
    
    result = calculate_lines_and_curves( img, preprocess, canny, normal, full, line, args.debug )
    
    result[1].save(args.output.name)
    print json.dumps(result[0])

def main():
    
    parser = argparse.ArgumentParser(description="""detect_lines: It finds a set of lines and curves in a picture
                                                    of an 3D printed object and adds them to the image. 
                                                    The output of the command is a JSON in the form 
                                                    {'curve': [], 'straight': [], 'other': []}
                                                    All of them are sets of points representing either the straight line,
                                                    the curve, or the unclassified line.""")
    
    parser.add_argument("--no-bilateral", action="store_const", const=True, default=False, help="Don't run the bilateral filter")
    parser.add_argument("--bilateral-diameter", default=5, help="The diameter of the bilateral filter", type=int)
    parser.add_argument("--bilateral-sspace", default=50, help="The sigma color of the bilateral filter", type=int)
    parser.add_argument("--bilateral-scolor", default=50, help="The sigma space of the bilateral filter", type=int)
    
    parser.add_argument("--no-normal",  action="store_const", const=True, default=False, help="Don't remove edges with the normal distribution filter")
    parser.add_argument("--normal-cut", default=99, type=float, help="The cut point to remove edges by size")
    
    parser.add_argument("--no-full",  action="store_const", const=True, default=False, help="Don't remove edges with the full edge filter")
    parser.add_argument("--full-cut", default=1, type=float, help="The cut ratio between the sum of the smaller edges and the larger edge")
    
    parser.add_argument("--canny-minthresh", default=20, type=int, help="Lower threshold of the Canny filter")
    parser.add_argument("--canny-ratio",     default=6, type=float, help="Ratio between the lower and upper threshold of the canny filter")
    
    parser.add_argument("--line-window", default=40, type=int,help= "Window to search for the line to fit with curve or straight line")
    parser.add_argument("--line-ratio", default=1.5, type=float, help="Number of times curve fit should be better than straight line fit")
    parser.add_argument("--line-maxfit", default=1, type=float, help="Maxmimum least squares average to declare a fit")
    
    parser.add_argument("--debug", action="store_const", const=True, default=False, help="Opens a window at each step showing what's happening with the algorithm")
    
    parser.add_argument("input", type=argparse.FileType('r'), help="input image file")
    parser.add_argument("output", type=argparse.FileType('w'), help="output image file")
    
    args = parser.parse_args()
    
    execute_task(args)
    
if __name__ == "__main__":
    main()

