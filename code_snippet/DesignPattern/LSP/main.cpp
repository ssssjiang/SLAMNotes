#include "Gun.h"
#include "Soldier.h"

using namespace bluefish;

int main()
{
    // Soldier soldier;
    // soldier.SetGun(new Rifle());
    // soldier.KillEnemy();
    Snipper snipper;
    snipper.KillEnemy(new AUG());
}