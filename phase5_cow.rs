//! Phase 5 copy-on-write runtime helpers.

#[derive(Clone, Debug)]
pub struct CowBudget {
    pub max_clones_per_tick: usize,
}

impl Default for CowBudget {
    fn default() -> Self {
        Self {
            max_clones_per_tick: 128,
        }
    }
}

#[inline]
pub fn should_clone(current_clones: usize, budget: &CowBudget) -> bool {
    current_clones < budget.max_clones_per_tick
}
