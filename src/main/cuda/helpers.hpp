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
  __device__ unsigned int manhattanDistanceTo (const Point<T>& b){
    return abs(x-b.x)+abs(y-b.y);
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



class ColorsARGB{
public:
  static constexpr const unsigned int BLACK = 0xff000000;
  static constexpr const unsigned int WHITE = 0xffffffff;
  static constexpr const unsigned int PINK  = 0xffb469ff;
  static constexpr const unsigned int YELLOW= 0xff00ffff;
  static constexpr const unsigned int GOLD  = 0xff00d7ff;
}; 

struct Color{
  char r;
  char g;
  char b;
  char a;
};
