#ifndef HELPERS
#define HELPERS

#define DEBUG_MODE
#ifdef DEBUG_MODE 
  #define ASSERT(x) assert(x)
#else 
  #define ASSERT(x) do {} while(0)
#endif

#ifndef CUDART_VERSION
  #error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 9000) //for cuda 9 and later, use __any_sync(__activemask(), predicate) instead, see Programming guide, B.13 for more details
  #define __ALL(predicate) __all_sync(__activemask(), predicate)
  #define __ANY(predicate) __any_sync(__activemask(), predicate)
#else
  #define __ALL(predicate) __all(predicate)
  #define __ANY(predicate) __any(predicate)
#endif


template <class T> struct Point {
  T x;
  T y;
  __device__ Point(){}
  __device__ Point(const T x,const T y) : x(x), y(y){} 
  __device__ Point(const T both) : x(both), y(both){} 
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
  __device__ Point<T> operator% (const Point<T>& b){
    return Point<T>(x%b.x, y%b.y);
  }
    __device__ const Point<T> operator+ (const Point<T>& b) const{
    return Point<T>(x+b.x, y+b.y);
  }
  __device__ const Point<T> operator- (const Point<T>& b) const{
    return Point<T>(x-b.x, y-b.y);
  }
  __device__ const Point<T> operator* (const Point<T>& b) const{
    return Point<T>(x*b.x, y*b.y);
  }
  __device__ const Point<T> operator/ (const Point<T>& b) const{
    return Point<T>(x/b.x, y/b.y);
  }
  __device__ const Point<T> operator% (const Point<T>& b) const{
    return Point<T>(x%b.x, y%b.y);
  }
  // __device__ unsigned int manhattanDistanceTo (const Point<T>& b){
  //   return abs(x-b.x)+abs(y-b.y);
  // }
  __device__ float distanceTo (const Point<T>& b){
  return sqrtf((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y));
  }
  template <class S> __device__ Point<S> cast() {
    return Point<S>((S)x, (S)y);
  }
  template <class S> __device__ const Point<S> cast() const {
    return Point<S>((S)x, (S)y);
  }
}; 

template <class T> struct Rectangle {
  Point<T> left_bottom;
  Point<T> right_top;
  __device__ Rectangle(){}
  __device__ Rectangle(Point<T> lt, Point<T> rb) : left_bottom(lb), right_top(rt){}
  __device__ Rectangle(const T lbx,const T lby, const T rtx,const T rty) : left_bottom(Point<T>(lbx, lby)), right_top(Point<T>(rtx, rty)){} 
  __device__ ~Rectangle(){}
  __device__ Point<T> const getLeftTop(){ return Point<T>(left_bottom.x, right_top.y); }
  __device__ Point<T> const getRightBottom(){ return Point<T>(right_top.x, left_bottom.y); }
  __device__ Point<T> const size(){ return right_top - left_bottom;}
};

struct pixel_info_t{
  unsigned int value;
  float weight;
};

struct color_t
{
  union
  {
     char r;
     char g;
     char b;
     char a;
  };
  unsigned int intValue;
};
     


class ColorsRGBA{ 
public:
//when human read, it is abgr, because of endianity
//                                              aabbggrr
  static constexpr const unsigned int BLACK = 0xff000000;
  static constexpr const unsigned int WHITE = 0xffffffff;
  static constexpr const unsigned int PINK  = 0xffb469ff;
  static constexpr const unsigned int YELLOW= 0xff00ffff;
  static constexpr const unsigned int GOLD  = 0xff00d7ff;
  static constexpr const unsigned int BLUE  = 0xffff0000;
  static constexpr const unsigned int GREEN = 0xff00ff00;
  static constexpr const unsigned int RED   = 0xff0000ff;
}; 

#endif
