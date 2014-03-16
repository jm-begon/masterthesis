# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of factory function to help create usual cases of coordinator
"""


from FilterGenerator import FilterGenerator, Finite3SameFilter
from Convolver import RGBConvolver
from SubWindowExtractor import (MultiSWExtractor, SubWindowExtractor,
                                FixTargetSWExtractor)
from NumberGenerator import OddUniformGenerator, NumberGenerator
from FeatureExtractor import ImageLinearizationExtractor
from Aggregator import AverageAggregator
from ConvolutionalExtractor import ConvolutionalExtractor
from Coordinator import RandConvCoordinator, PixitCoordinator
from TaskManager import ParallelCoordinator


__all__ = ["coordinatorRandConvFactory", "coordinatorPixitFactory"]


def coordinatorPixitFactory(
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        fixedSize=False,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        nbJobs=-1, verbosity=10, tempFolder=None):
    """
    Factory method to create :class:`PixitCoordinator`

    Parameters
    ----------
    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    fixedSize : boolean (default : False)
        Whether to use fixe size subwindow. If False, subwindows are drawn
        randomly. If True, the target size is use as the subwindow size and
        only the position is drawn randomly
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    Return
    ------
        coordinator : :class:`Coordinator`
            The PixitCoordinator (possibly decorated) corresponding to the set
            of parameters
    Notes
    -----
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    #SubWindowExtractor
    swNumGenerator = NumberGenerator()
    if fixedSize:
        swExtractor = FixTargetSWExtractor(subwindowTargetWidth,
                                           subwindowTargetHeight,
                                           subwindowInterpolation,
                                           swNumGenerator)
    else:
        swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                         subwindowMaxSizeRatio,
                                         subwindowTargetWidth,
                                         subwindowTargetHeight,
                                         subwindowInterpolation,
                                         swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, True)

    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #COORDINATOR
    coordinator = PixitCoordinator(multiSWExtractor, featureExtractor)

    if nbJobs == 1 and verbosity <= 0:
        return coordinator
    return ParallelCoordinator(coordinator, nbJobs, verbosity, tempFolder)


def coordinatorRandConvFactory(
        nbFilters=5,
        filterMinVal=-1, filterMaxVal=1,
        filterMinSize=1, filterMaxSize=17,
        filterNormalisation=FilterGenerator.NORMALISATION_MEANVAR,
        aggregatorNeighborhoodWidth=3, aggregatorNeighbordhoodHeight=3,
        nbSubwindows=10,
        subwindowMinSizeRatio=0.5, subwindowMaxSizeRatio=1.,
        subwindowTargetWidth=16, subwindowTargetHeight=16,
        subwindowInterpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        includeOriginalImage=False,
        nbJobs=-1, verbosity=10, tempFolder=None):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nbFilters : int >= 0 (default : 5)
        The number of filter
    filterMinVal : float (default : -1)
        The minimum value of a filter component
    filterMaxVal : float : filterMinVal <= filterMaxVal (default : 1)
        The maximum value of a filter component
    filterMinSize : int >= 0 : odd number (default : 1)
        The minimum size of a filter
    filterMaxSize : int >= 0 : odd number s.t.  filterMinSize <= filterMaxSize
    (default : 1)
        The maximum size of a filter
    filterNormalisation : int (default : FilterGenerator.NORMALISATION_MEANVAR)
        The filter normalisation policy. See also :class:`FilterGenerator`

    aggregatorNeighborhoodWidth : int > 0 (default : 3)
        The neigborhood width of the aggregation
    aggregatorNeighbordhoodHeight : : int > 0 (default : 3)
        The neigborhood height of the aggregation

    nbSubwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    subwindowMinSizeRatio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowMaxSizeRatio : float : subwindowMinSizeRatio
    <= subwindowMaxSizeRatio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    subwindowTargetWidth : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    subwindowTargetHeight : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    subwindowInterpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    includeOriginalImage : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    nbJobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    tempFolder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator (possibly decorated) corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Aggregator
        Base instance of :class:`AverageAggregator`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    filterValGenerator = NumberGenerator(filterMinVal, filterMaxVal)
    filterSizeGenerator = OddUniformGenerator(filterMinSize, filterMaxSize)
    baseFilterGenerator = FilterGenerator(filterValGenerator,
                                          filterSizeGenerator,
                                          normalisation=filterNormalisation)
    filterGenerator = Finite3SameFilter(baseFilterGenerator, nbFilters)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    aggregator = AverageAggregator(aggregatorNeighborhoodWidth,
                                   aggregatorNeighbordhoodHeight,
                                   subwindowTargetWidth, subwindowTargetHeight)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator()
    swExtractor = SubWindowExtractor(subwindowMinSizeRatio,
                                     subwindowMaxSizeRatio,
                                     subwindowTargetWidth,
                                     subwindowTargetHeight,
                                     subwindowInterpolation, swNumGenerator)

    multiSWExtractor = MultiSWExtractor(swExtractor, nbSubwindows, False)

    #ConvolutionalExtractor
    convolutionalExtractor = ConvolutionalExtractor(filterGenerator,
                                                    convolver,
                                                    multiSWExtractor,
                                                    aggregator,
                                                    includeOriginalImage)
    #FEATURE EXTRACTOR
    featureExtractor = ImageLinearizationExtractor()

    #COORDINATOR
    coordinator = RandConvCoordinator(convolutionalExtractor, featureExtractor)
    if nbJobs == 1 and verbosity <= 0:
        return coordinator
    return ParallelCoordinator(coordinator, nbJobs, verbosity, tempFolder)







if __name__ == "__main__":

    from CifarLoader import CifarLoader
    from ImageBuffer import ImageBuffer

    path = "data_batch_1"
    imgBuffer = CifarLoader(path, outputFormat=ImageBuffer.NUMPY_FORMAT)

    #coord = coordinatorRandConvFactory(verbosity=50)
    coord = coordinatorRandConvFactory(nbJobs=1, verbosity=0)

    X, y = coord.process(imgBuffer[0:10])
    print X.shape, len(y)
