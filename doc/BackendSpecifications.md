# BackendSpecifications

## SBE : SIMD BACKEND
- index1
    - op+ op- op* op/ 
- index2
    - op+ op- op* op/
- float3
    - op+ op- op* op/
- float2
    - op+ op- op* op/

- kernelInfo

//Scheduler sollte das regeln
/*
- index3 GlobalSize()
    - op+ op- op* op/
- index3 GlobalIdx()
    - op+ op- op* op/
- index3 LocalSize()
    - op+ op- op* op/
- index3 LocalIdx()
    - op+ op- op* op/
*/

## Datatypes


- Array<T>
    - T [*index* i]  



- 2DMatrix<T,Dim>
    -T[Dim] [*index2* i]  

//Padding
- TextureRGB
    - float3 [*index2* i]

//Padding
- TextureGrayScale
    - float [*index2* i]



## Iterators

- 2DRectangleRange(index2 start, index2 end)
    -2DRectangleIterator begin
    -2DRectangleIterator end

- 2DRectangleIterator(index2 start, index2 end)
    - operator++
    - index2 operator*

//geht die Daten so durch, wie sie im Speicher liegen
- NativeRange<T>(T obj)
    -  NativeIterator<T> begin
    -  NativeIterator<T> end
- NativeIterator<T>(T start, T end)
    - operator++
    - T operator*
    - 
## Buffer

- ArrayBuffer<T> cacheAllocArray(_Size)

- 2DMatrixBuffer<T> cacheAlloc2DMatrix<_Width,_Heigth>()

- ArrayBuffer<T>
    - T [*index* i] //local

- 2DMatrixBuffer<T,Dim>
    -T[Dim] [*index2* i]  //local

## Schedulars

- gridStripSchedular(kernelInfo,int itemCount,
 lambda(index idx,args...), args...)

- gridStripSchedular2D(kernelInfo,index2 GlobalStart,index2 GlobalEnd
 lambda(index2 idx,args...), args...)

- tilesSchedular2D(kernelInfo,index2 GlobalStart, index2 GlobalEnd,index2 tilesSizePerBlock, index2 Overlap, args...
 lambda(index2 idx,args...))


