#include <stdio.h>
#include <fenv.h>

int main() {
    fesetround(FE_DOWNWARD);
    float x = 1.9f + 1.0f;
    printf("%f\n", x);
}