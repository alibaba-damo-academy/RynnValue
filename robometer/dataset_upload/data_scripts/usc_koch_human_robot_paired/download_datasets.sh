#!/bin/bash
source ./venv/bin/activate

datasets=(
    "usc_koch_throw_the_orange_cup_away_red_trash_can"
    "usc_koch_suboptimal_throw_the_orange_cup_away_red_trash_can"
    "usc_koch_failure_throw_the_orange_cup_away_red_trash_can"
    "usc_koch_throw_the_black_marker_away_blue_trash_can"
    "usc_koch_suboptimal_throw_the_black_marker_away_blue_trash_can"
    "usc_koch_failure_throw_the_black_marker_away_blue_trash_can"
    "usc_koch_open_the_red_trash_bin_red_trash_bin"
    "usc_koch_suboptimal_open_the_red_trash_bin_red_trash_bin"
    "usc_koch_failure_open_the_red_trash_bin_red_trash_bin"
    "usc_koch_open_the_green_trash_bin_green_trash_bin"
    "usc_koch_suboptimal_open_the_green_trash_bin_green_trash_bin"
    "usc_koch_failure_open_the_green_trash_bin_green_trash_bin"
    "usc_koch_open_the_blue_trash_bin_blue_trash_bin"
    "usc_koch_suboptimal_open_the_blue_trash_bin_blue_trash_bin"
    "usc_koch_failure_open_the_blue_trash_bin_blue_trash_bin"
    "usc_koch_separate_the_red_and_orange_and_orange_cups"
    "usc_koch_suboptimal_separate_the_red_and_orange_and_orange_cups"
    "usc_koch_failure_separate_the_red_and_orange_and_orange_cups"
    "usc_koch_separate_the_purple_and_orange_and_orange_cups"
    "usc_koch_suboptimal_separate_the_purple_and_orange_and_orange_cups"
    "usc_koch_failure_separate_the_purple_and_orange_and_orange_cups"
    "usc_koch_separate_the_purple_and_red_and_red_cups"
    "usc_koch_suboptimal_separate_the_purple_and_red_and_red_cups"
    "usc_koch_failure_separate_the_purple_and_red_and_red_cups"
    "usc_koch_move_the_orange_cup_from_right_to_left"
    "usc_koch_suboptimal_move_the_orange_cup_from_right_to_left"
    "usc_koch_failure_move_the_orange_cup_from_right_to_left"
    "usc_koch_move_the_orange_cup_from_left_to_right"
    "usc_koch_suboptimal_move_the_orange_cup_from_left_to_right"
    "usc_koch_failure_move_the_orange_cup_from_left_to_right"
)

for dataset in ${datasets[@]}; do
    hf download abraranwar/${dataset} --repo-type=dataset --local-dir=./datasets/usc_koch_human_robot_paired/robot/${dataset}
done


gdown --fuzzy https://drive.google.com/file/d/1VsV9GN784WGmuNYtMeIduXfk72WSaS1E/view?usp=drive_link
unzip recordings.zip -d ./datasets/usc_koch_human_robot_paired/human
rm recordings.zip
