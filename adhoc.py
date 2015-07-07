"""
Ad-hoc synthetic data generation process.

"""
import os
import time
import logging
import argparse
import datetime

import numpy as np
import pandas as pd


def tsnow():
    return datetime.datetime.fromtimestamp(time.time())\
                   .strftime("%Y-%d-%H-%M-%S")


def adhoc_gen(n, m, nterms):
    """Generative an n x m (student, course) matrix of data. The actual data
    will not have all entries filled. Which entries are filled is determined
    based on random draws for each term. The more terms, the more data there
    will be. Use an ad-hoc generation process to produce the following
    features:

        sid
        cid
        age
        gender
        alevel
        clevel
        chrs
        grade
        hsgpa
        cum_gpa
        cum_cgpa
    """
    logging.info(
        "generating data for:\n%d\tstudents\n%d\tcourses\n%d\tterms" % (
            n, m, nterms))

    courses = pd.DataFrame()
    students = pd.DataFrame()
    dyads = pd.DataFrame()
    lo, hi = (0, 4)

    # Draw student data.
    logging.info("drawing student data")
    students['gender'] = np.random.binomial(1, 0.5, n)  # 0 is male, 1 female
    students['age'] = np.random.normal(23, 2, n).astype(np.int)
    # TODO: randomly add some older folks.

    # Randomly draw academic level and use that to calculate credit hours taken.
    students['alevel'] = np.random.multinomial(
        1, (0.31, 0.32, 0.21, 0.14, 0.02), n).nonzero()[-1]
    students['schrs'] = students['alevel'] * 30
    bin_alevel = lambda s: \
        pd.cut(s, bins=[0, 30, 60, 90, 120, 200], include_lowest=True,
               labels=range(5)).astype(np.int)

    key = 'hsgpa'
    students[key] = np.random.normal(3.0, 0.5, n)
    students.loc[students[key] < lo, key] = lo
    students.loc[students[key] > hi, key] = hi
    students['cum_gpa'] = students[key]  # set initial cumulative gpa to hsgpa

    # Draw course data
    logging.info("drawing course data")
    courses['chrs'] = np.random.poisson(4, m)
    courses['clevel'] = np.random.multinomial(
        1, (0.0000, 0.4570, 0.1960, 0.2640, 0.0790, 0.0021, 0.0019), m)\
            .nonzero()[-1]

    # Now let's randomly figure out how many people have taken this course so
    # far. Let's imagine we have 3 types of courses: (1) new courses
    # (2) courses which have been around for a few semesters, and (3) courses
    # which have been around for a long time. First we choose about 5% as being
    # new. Then we draw the others from a multivariate normal.
    prop_new = 0.03
    nnew = int(m * prop_new)
    nold = m - nnew
    means = [20, 100, 300]
    ngroups = len(means)
    cov = np.eye(ngroups) * 100
    ndraws = int(np.ceil(nold / float(ngroups)))
    vec = np.random.multivariate_normal(means, cov, ndraws)\
            .reshape(ngroups * ndraws)[:nold]
    vec[vec < 0] = 0
    nenrolled = np.concatenate((vec, [0] * nnew)).astype(np.int)
    np.random.shuffle(nenrolled)
    courses['total_enrolled'] = nenrolled

    # Draw total cumulative GPA of all students who have taken the course.
    key = 'cum_cgpa'
    courses[key] = np.random.normal(3.0, 0.5, m)
    courses.loc[courses[key] < lo, key] = lo
    courses.loc[courses[key] > hi, key] = hi

    # Select dyads randomly for each term, constrained by clevel/alevel.
    course_by_clevel = \
        courses.groupby('clevel')\
               .apply(lambda df: df.index.values)\
               .reindex([1, 2, 3, 4, 5, 6])\
               .apply(lambda val: [] if isinstance(val, float) else val)

    logging.info("drawing dyads for each term")
    for t in xrange(nterms):
        logging.info("drawing dyads for term %d" % t)
        ncourses = (np.random.power(3, n) * 5).astype(np.int)
        maxnc = ncourses.max()
        factors = np.random.multinomial(1, (0.7, 0.2, 0.1), (n, maxnc))\
                    .nonzero()[-1]
        offsets = factors * (-1 * np.random.binomial(1, 0.5, n*maxnc))
        clevels = (offsets.reshape(n, maxnc) +
                   students['alevel'].reshape(n, 1) + 1)
        clevels[clevels < 1] = 1

        # Select courses based on clevel values.
        _dyads = []
        for s in range(n):
            clevs = clevels[s,:ncourses[s]]
            for clevel, count in pd.Series(clevs).value_counts().iterkv():
                options = course_by_clevel[clevel]
                count = len(options) if count > len(options) else count
                if not count:
                    continue

                choices = np.random.choice(options, count, replace=False)
                new_dyads = zip(*([s]*count, choices, [t]*count))
                _dyads.extend(new_dyads)

        _dyads = pd.DataFrame(_dyads, columns=['sid', 'cid', 'term'])\
                   .merge(students, how='left', left_on='sid',
                          right_index=True)\
                   .merge(courses, how='left', left_on='cid', right_index=True)

        # How many credit hours is each student taking this term?
        _dyads['term_chrs'] = _dyads.groupby('sid')['chrs'].transform('sum')
        _dyads['term_enrolled'] = _dyads.groupby('cid')['chrs']\
                                        .transform('count')

        # Calculate grades for courses.
        # cum_gpa * cum_cgpa + age/30 + gender(0.2) + (alevel - 1)(clevel - 1)
        # - [(chrs / 3) - 1]
        logging.info("calculating grades for courses selected")
        _dyads['grade'] = (
            # grades so far is the biggest contributor.
            (_dyads['cum_gpa'] * _dyads['cum_cgpa']) +
            _dyads['cum_gpa'] + _dyads['cum_cgpa'] +
            # older students perform just slightly better
            _dyads['age'] / 30 +
            # let's say female students perform slightly better
            _dyads['gender'] * 0.2 -
            # match up between alevel and clevel should improve grade
            abs(_dyads['alevel'] - _dyads['clevel']) -
            # smaller number of credit hours should be easier
            (_dyads['chrs'] / 3. - 1) -
            # less classes in a term should be easier
            (_dyads['term_chrs'] - 12) * 0.5 -
            # smaller class sizes should improve grades
            _dyads['term_enrolled'] * 0.01
        )
        logfunc = lambda g: 4 / (1 + np.e ** -(g - 6))
        _dyads['grade'] = _dyads['grade'].apply(logfunc)

        # Randomly decrease some of the higher grades to simulate outside
        # effects. May need tweaking to get reasonable grade distribution.
        high_grades = _dyads['grade'][_dyads['grade'] > 3.5]
        bad = np.random.normal(3.5, 0.1, int(high_grades.shape[0] * 0.04))
        mid = np.random.normal(1.5, 0.1, int(high_grades.shape[0] * 0.15))
        low = np.random.normal(0.5, 0.1, int(high_grades.shape[0] * 0.15))
        effects = np.concatenate((bad, mid, low))
        np.random.shuffle(effects)
        affected = np.random.choice(
            high_grades.index, replace=False, size=len(effects))
        _dyads.loc[affected, 'grade'] = _dyads.loc[affected, 'grade'] - effects
        _dyads.loc[_dyads['grade'] < lo] = lo
        _dyads.loc[_dyads['grade'] > hi] = hi

        # Update attributes and add new dyads.
        logging.info("updating attributes based on course selections")

        # Update cum_gpa.
        # For the cumulative student GPA, we need to figure out how many quality
        # points each student attempted this semester and how many were earned.
        term_chrs = _dyads.groupby('sid')['term_chrs'].first()
        term_qpts = _dyads.groupby('sid')\
                           .apply(lambda df: (df['grade'] * df['chrs']).sum())\
                           .reindex(students.index)\
                           .fillna(0)\
                           .sort_index()
        prev_qpts = (students['schrs'] * students['cum_gpa']).sort_index()
        students['schrs'] = (
            students['schrs'] +
            term_chrs.reindex(students.index).fillna(0).sort_index()
        )
        qpts_possible = students['schrs'] * 4

        key = 'cum_gpa'
        students[key] = ((term_qpts + prev_qpts) / qpts_possible) * 4
        students.loc[students[key].isnull(), key] = students['hsgpa']

        # Update alevel from student chrs.
        students['alevel'] = bin_alevel(students['schrs'])

        # Update cum_cgpa.
        # The course cumulative GPA is the grade of all students who have taken
        # the course normalized using the number of enrollments historically.
        term_grade_sum = _dyads.groupby('cid')['grade'].transform('sum')
        _dyads['term_cgpa'] = term_grade_sum / _dyads['term_enrolled']

        cgpa_sums = (
            courses['cum_cgpa'] * courses['total_enrolled'] +
            term_grade_sum.reindex(courses.index).fillna(0).sort_index()
        )
        courses['total_enrolled'] = (
            courses['total_enrolled'] +
            _dyads.groupby('cid')['term_enrolled'].first()\
                    .reindex(courses.index)\
                    .fillna(0).sort_index()
        )
        courses['cum_cgpa'] = cgpa_sums / courses['total_enrolled']

        dyads = pd.concat([dyads, _dyads])

    logging.info("generated {:,} dyads in total".format(len(dyads)))
    return dyads


def make_parser():
    parser = argparse.ArgumentParser(
        description='generate synthetic university grade data')
    parser.add_argument(
        '-n', '--nstudents', type=int,
        help='number of students')
    parser.add_argument(
        '-m', '--ncourses', type=int,
        help='number of courses')
    parser.add_argument(
        '-t', '--nterms', type=int,
        help='number of enrollment terms')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False)
    parser.add_argument(
        '-o', '--output',
        action='store_true', default=False)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="[%(asctime)s]: %(message)s")

    data = adhoc_gen(args.nstudents, args.ncourses, args.nterms)
    if args.output:
        fname = 'data-n%d-m%d-t%d-d%d-%s.csv' % (
            args.nstudents, args.ncourses, args.nterms, len(data), tsnow())
        logging.info('writing data to: %s' % fname)
        data.to_csv(fname, index=False)
