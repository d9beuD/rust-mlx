//! A scoped graph-construction mode for decode-equivalent multi-token verification.
use std::cell::Cell;
thread_local! {static ACTIVE:Cell<bool>=const{Cell::new(false)};}
thread_local! {static ROWS:Cell<bool>=const{Cell::new(false)};}
pub fn rows() -> bool {
    ROWS.with(Cell::get)
}
pub fn with_rows<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ROWS.with(|x| x.set(self.0));
        }
    }
    let _restore = Restore(ROWS.with(|x| x.replace(true)));
    f()
}
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}
pub fn with_mode<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE.with(|x| x.set(self.0));
        }
    }
    let old = ACTIVE.with(|x| x.replace(true));
    let _restore = Restore(old);
    with_rows(f)
}
