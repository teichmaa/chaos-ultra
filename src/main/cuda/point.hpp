template <class T> struct Point {
  T x;
  T y;
  __device__ Point(){}
  __device__ Point(const T x,const T y) : x(x), y(y){} 
  __device__ ~Point(){}
  __device__ Point<T> operator+ (const Point<T>& b){
    return Point<T>(x+b.x, y+b.y);
  }
  __device__ Point<T> operator- (const Point<T>& b){
    return Point<T>(x-b.x, y-b.y);
  }
  __device__ Point<T> operator* (const Point<T>& b){
    return Point<T>(x*b.x, y*b.y);
  }
  __device__ Point<T> operator/ (const Point<T>& b){
    return Point<T>(x/b.x, y/b.y);
  }
}; 
