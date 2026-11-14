//! Video analysis: find every person, follow them, and gather evidence about
//! who they are. Produces tracks; `decide` turns them into a blur plan.

pub mod cuts;
pub mod detect;
pub mod frames;
pub mod mask;
pub mod track;
