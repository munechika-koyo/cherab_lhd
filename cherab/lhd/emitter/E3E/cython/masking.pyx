# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This module is optimized for EMC3-EIRINE in LHD by Koyo Munechika.

cimport cython
from cherab.lhd.emitter.E3E.cython cimport IntegerFunction3D
from libc.limits cimport INT_MIN


cdef class EMC3Mask(IntegerFunction3D):
    """
    Masking EMC3-EIRINE grid space to identify whether or not a mesh exists at the point.
    This instance is callable function returning 1 corresponding in 
    3D space where EMC3's index function returns a physical index, otherwise 0.

    :param IntegerFunction3D index_func: EMC3's index_funcion returning a physical index.
    """

    def __init__(self, object index_func not None):

        # validate arguments
        if not callable(index_func):
            raise TypeError("This function is not callable.")
    
        # populate internal attributes
        self._index_func = index_func

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int evaluate(self, double x, double y, double z) except? INT_MIN:
        
        cdef:
            int index

        index = self._index_func(x, y, z)

        if index < 0:
            return 0
        else:
            return 1