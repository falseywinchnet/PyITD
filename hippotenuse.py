from mpmath import *
mp.dps = 500  # set precision

'''
             ..ed$$$$$$$$$$$$$be      *F..
       ^   z$$$$$$$$$$$$$$$$$$$$$$$$$$$$
          $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$.
         $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*e.
        4$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$be$$$$$$c
        4$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$L
        ^$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
         ^*$$$$$$$$$$$$$$$$$$$$$$$$$$$F^*$$$$$$$$$%
           ^"$$$$$$$$$$$$$$$$$$$$$$$$$    "**$P*"  Gilo94'
             4$$$F"3$$$$       4$$$$
             d$$$$ 4$$$$       4$$$$                 Hippo
                                                     ^^^^^
'''   
#with this function we provide the closed form algebraic expressions for deriving the hypotenuse of a triangle
#inscribing a unit circle of radius 1, where the triangle series begins by inscribing three points marked as the center,
#the point on the circle at 0, and the point on the circle at 90. That value can be determined empirically to be sqrt(2).
#for each value of n higher  than this we derive the identity of the triangle as a triangle with two sides of 1 and a rapidly 
#diminishing arc which divides in half every time.
#Going in the other direction, we may derive a value which could use the hypotenuse
#of the previous triangle and a rapidly diminishing but known a and b angle, culminating in n=2 with the last valid triangle.
#n=1 is for a triangle with zero area. n=0 is for a triangle that encapsulates a 360 degree circle...
#this is purely a hypothetical projection as we have no other criteria which may be used to determine their correct identity.
#however, we observe cosecant(pi/16) = 4 and cosecant(pi/32) = 5.cosecant(pi/8)=6, cosecant(pi/4)=1.
#other criteria that could be used is a further progession of the cosecant series but this quickly explodes to exponential values.
#the largest valid value of this projection series is (64 * (2 + sqrt(2 + sqrt(2 + sqrt(2))))**2 * (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**2)/(2 - sqrt(2 + sqrt(2)))**2
#that is the hypotenuse, and the two sides are (4 *(2 + sqrt(2 + sqrt(2 + sqrt(2)))) *sqrt(4* (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**2 + (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**(5/2)))/(2 - sqrt(2 + sqrt(2)))
#the degrees are 0, 0, and 180.thereinlies the hippotenuse. i reject all alternatives as utterly unimaginative-
#the hippotenuse deserves extensive study for his unique and interesting qualities.
#alternatively we can use the simple cosine rule which is what i have done here.


def generate_hypotenuse(n):
    if n = 1:
      return 2
    elif n == 2:
      a = sqrt(2 + sqrt(2 + sqrt(2+ sqrt(2 + sqrt(2 + sqrt(2))))))
      return a
    elif(n==3):
      a = sqrt(2 + sqrt(2 + sqrt(2+ sqrt(2 + sqrt(2)))))

      return a
    elif(n==4):
      a = sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2))))
      return a
    elif n == 5:
      a  = sqrt(2 + sqrt(2 + sqrt(2)))
      return a 
    elif n==  6:
      return sqrt(2 + sqrt(2)))
    elif n == 7:
        return sqrt(2) #90 degrees
    elif n == 8:
        return 1 - sqrt(2)
    elif n == 9:
        return  1 - sqrt(2 + sqrt(2))
    else:
      q = n - 8
      x = 2
      for _ in range(q):
          x = 2 + sqrt(x)
      return 1 - sqrt(x)



#this yields some interesting consequences:
def hypotenuse(x, n):
  value = 2
  for _ in range(abs(n)):
       value = 2 + sqrt(value)
  if n <0:
      value =2 * (1-  sqrt(value)/2)
  else:
      value =  sqrt(value)
  return x * value
  #if we assess that n is the degree, and we 
