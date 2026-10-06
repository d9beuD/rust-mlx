//! Experimental policies act on draft efficiency only; the target always verifies fully.
use std::collections::VecDeque;

/// Calibration from the recorded target HC/QMV depth sweep, seconds per complete round.
pub const TARGET_ROUND_COSTS: [f64; 3] = [0.031545832, 0.038575983, 0.046995903];

pub(crate) struct DepthPolicy {
    maximum: usize,
    adaptive: bool,
    costs: [f64; 3],
    recent: VecDeque<(usize, usize)>,
}
impl DepthPolicy {
    pub(crate) fn new(maximum: usize, adaptive: bool, costs: [f64; 3]) -> Self {
        Self {
            maximum,
            adaptive,
            costs,
            recent: VecDeque::new(),
        }
    }
    pub(crate) fn choose(&self) -> usize {
        if !self.adaptive {
            return self.maximum;
        }
        let maximum = self.maximum.min(3);
        let priors = [0.82, 0.77, 0.68];
        let mut expectation = 1.0;
        let mut survival = 1.0f64;
        let mut rates = [0.0; 3];
        for j in 1..=maximum {
            let trials = self.recent.iter().filter(|(k, _)| *k >= j).count();
            let successes = self
                .recent
                .iter()
                .filter(|(k, a)| *k >= j && *a >= j)
                .count();
            survival =
                survival.min((4.0 * priors[j - 1] + successes as f64) / (4.0 + trials as f64));
            expectation += survival;
            rates[j - 1] = expectation / self.costs[j - 1];
        }
        let mut best = maximum;
        // Prefer the established maximum unless the predicted rate improves by3%.
        for k in 1..maximum {
            if rates[k - 1] > rates[best - 1] * 1.03 {
                best = k;
            }
        }
        best
    }
    pub(crate) fn observe(&mut self, proposed: usize, accepted: usize) {
        if self.adaptive {
            self.recent.push_back((proposed, accepted));
            if self.recent.len() > 16 {
                self.recent.pop_front();
            }
        }
    }
}

/// Rebuild on repeated rejection, briefly use the complete head, then try the shortlist.
#[derive(Default)]
pub(crate) struct VocabularyPolicy {
    misses: usize,
    full_rounds: usize,
    last_refresh: usize,
}
impl VocabularyPolicy {
    pub(crate) fn use_full(&self) -> bool {
        self.full_rounds > 0
    }
    pub(crate) fn observe(&mut self, accepted: usize, round: usize) -> bool {
        if self.full_rounds > 0 {
            self.full_rounds -= 1;
            self.misses = 0;
            return false;
        }
        self.misses = if accepted == 0 { self.misses + 1 } else { 0 };
        if self.misses >= 2 && round.saturating_sub(self.last_refresh) >= 8 {
            self.last_refresh = round;
            self.full_rounds = 4;
            self.misses = 0;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_reduces_wasted_depth_and_keeps_high_acceptance_depth() {
        let mut p = DepthPolicy::new(3, true, TARGET_ROUND_COSTS);
        assert_eq!(p.choose(), 3);
        for _ in 0..16 {
            p.observe(3, 0);
        }
        assert_eq!(p.choose(), 1);
        for _ in 0..16 {
            p.observe(3, 3);
        }
        assert_eq!(p.choose(), 3);
        assert_eq!(DepthPolicy::new(7, false, TARGET_ROUND_COSTS).choose(), 7);
        assert!(DepthPolicy::new(7, true, TARGET_ROUND_COSTS).choose() <= 3);
    }
    #[test]
    fn cost_calibration_changes_choice_without_output_assumptions() {
        assert_eq!(DepthPolicy::new(3, true, [0.01, 0.1, 1.0]).choose(), 1);
        assert_eq!(DepthPolicy::new(2, true, TARGET_ROUND_COSTS).choose(), 2);
        let mut p = VocabularyPolicy::default();
        assert!(!p.observe(0, 7));
        assert!(p.observe(0, 8));
        for round in 9..13 {
            assert!(p.use_full());
            assert!(!p.observe(0, round));
        }
        assert!(!p.use_full());
        assert!(!p.observe(0, 13));
        assert!(!p.observe(0, 14));
        assert!(!p.observe(0, 15));
        assert!(p.observe(0, 16));
    }
}
