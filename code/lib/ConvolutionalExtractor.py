# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
""" """
import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

from NumpyToPILConvertor import NumpyPILConvertor


__all__ = ["ConvolutionalExtractor"]


class ConvolutionalExtractor:
    """
    ======================
    ConvolutionalExtractor
    ======================
    A :class:`ConvolutionalExtractor` extract features from images. It
    proceeds in 3 steps :

    1. Filtering
        It uses a :class:`FiniteFilter` to generate filters. Thoses filters
        are then applied by a :class:`Convolver` to the given image thus
        creating several new images (one per filter). Let us call them
        *image2*.
    2. Aggregation
        Each new *image2* is aggregated by a :class:`Aggregator`, yielding one
        smaller image (let us call them *image3*) by processed *image2*s.
    3. Subwindow extraction
        On each *image3* the same subwindows are extracted giving the set
        of *image4*. This set contains nb_filter*nb_subwindow images

    Note
    ----
    - Compatibily
        The :class:`FiniteFilter` and the :class:`Convolver` must be compatible
        with the kind of image provided !
        Example : the image is a RGB PIL image or RGB numpy array with a
        :class:`RGBConvolver` and a :class:`Finite3Filter`

    - Image representation
        See :mod:`ImageBuffer` for more information

    - It is also possible to include the original image in the process
    """

    def __init__(self, finiteFilter, convolver, multiSWExtractor, aggregator,
                 include_original_image=False):
        """
        Construct a :class:`ConvolutionalExtractor`

        Parameters
        ----------
        finiteFilter : :class:`FiniteFilter`
            The filter generator and holder
        convolver : :class:`Convolver`
            The convolver which will apply the filter. Must correspond with
            the filter generator and the image type
        aggregator : :class:`Aggregator`
            The aggregator which will aggregate after the filtering
        include_original_image : boolean (default : False)
            Whether or not to include the original image for the subwindow
            extraction part
        """
        self._finiteFilter = finiteFilter
        self._convolver = convolver
        self._swExtractor = multiSWExtractor
        self._aggregator = aggregator
        self._include_image = include_original_image

    def extract(self, image):
        """
        Extract feature from the given image

        Parameters
        ----------
        image : :class:`PIL.Image` or preferably a numpy array
            The image to process

        Return
        ------
        all_subwindow : a list of lists of subwindows
            The element e[i][j] is a numpy array correspond to the ith
            subwindow of the jth filter.
            If the original image is included, it correspond to the first
            (0th) filter.
        """

        #Converting image in the right format
        convertor = NumpyPILConvertor()
        image = convertor.pILToNumpy(image)

        filtered = []

        #Including the original image if desired
        if self._include_image:
            filtered.append(self._aggregator(image))
        #Applying the filters & Aggregating
        for filt in self._finiteFilter:
            #Filtering
            npTmp = self._convolver(image, filt)
            #Aggregating
            filtered.append(self._aggregator(npTmp))

        #Refreshing the boxes
        height, width, depth = filtered[0].shape
        self._swExtractor.refresh(width, height)

        #Extracting the subwindows
        nbFilters = len(self._finiteFilter)
        nbSubWindow = len(self._swExtractor)
        nbImageFactor = nbFilters
        if self._include_image:
            nbImageFactor += 1

        allSubWindows = [[0] * nbImageFactor for i in xrange(nbSubWindow)]

        for col, numpies in enumerate(filtered):
            #converting image to the right format
            img = convertor.numpyToPIL(numpies)
            #Extracting the subwindows s.s.
            subwindows = self._swExtractor.extract(img)
            for row in xrange(nbSubWindow):
                allSubWindows[row][col] = convertor.pILToNumpy(subwindows[row])

        return allSubWindows

if __name__ == "__main__":
    test = True
    if test:
        imgpath = "lena.png"
        from FilterGenerator import FilterGenerator, Finite3SameFilter
        from Convolver import RGBConvolver
        from SubWindowExtractor import MultiSWExtractor, SubWindowExtractor
        from NumberGenerator import OddUniformGenerator, NumberGenerator, IntegerUniformGenerator
        from Aggregator import AverageAggregator

        imgPil = Image.open(imgpath)
        img = np.array(imgPil)
        

        #CONVOLUTIONAL EXTRACTOR
        #Filter generator
        filterValGenerator = IntegerUniformGenerator(-5, 5)
        filterSizeGenerator = OddUniformGenerator(3,10)
        baseFilterGenerator = FilterGenerator(filterValGenerator, filterSizeGenerator)
        filterGenerator = Finite3SameFilter(baseFilterGenerator, 6)
        
        #Convolver
        convolver = RGBConvolver()
        
        #SubWindowExtractor
        subwindowTargetWidth = 200
        subwindowTargetHeight = 200
        swNumGenerator = NumberGenerator()
        swExtractor = SubWindowExtractor(0.5, 1., subwindowTargetWidth, subwindowTargetHeight, SubWindowExtractor.INTERPOLATION_BILINEAR, swNumGenerator)
        multiSWExtractor = MultiSWExtractor(swExtractor,4)
        
        #Aggregator
        aggregator = AverageAggregator(10, 10, subwindowTargetWidth , subwindowTargetHeight)
        
        convolutionalExtractor = ConvolutionalExtractor(filterGenerator, convolver, multiSWExtractor, aggregator)    
        
        res = convolutionalExtractor.extract(img)
        