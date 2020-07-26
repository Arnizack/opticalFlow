# BackendSpecifications

## SBE : SIMD BACKEND
- int
    - op+ op- op* op/

- int2
	- x
	- y
    - op+ op- op* op/

- int3
	- x
	- y
	- z
    - op+ op- op* op/

- int4
	- x
	- y
	- z
	- w
    - op+ op- op* op/
	
- float3
	- x
	- y
	- z
    - op+ op- op* op/
	
- float4
	- x
	- y
	- z
	- w
    - op+ op- op* op/
	
- float2
	- x
	- y
    - op+ op- op* op/

- kernelInfo

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


/*
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

*/
	
## Buffer


- ArrayBuffer<T>
    - T [*int* i] //local

- 2DMatrixBuffer<T,Dim>
    -T[Dim] [*int2* i]  //local
	
- TilesBuffer<T> 
    - T [*int2* i]

- ArrayBuffer<T> allocArrayBuffer(size)

- 2DMatrixBuffer<T> alloc2DMatrixBuffer(width,heigth)
	
- TilesBuffer<T> allocTilesBuffer(*kernelInfo* Kinfo, int2 dimensions, int2 tilesSize, int2 padding )

## Schedulars

- gridStripSchedular(kernelInfo,int itemCount,
 lambda(index idx,args...), args...)

//Row first order
- tilesSchedular2D(kernelInfo,index2 GlobalStart, index2 GlobalEnd,index2 tilesSizePerBlock, index2 Overlap, args...
 lambda(index2 idx,args...))


