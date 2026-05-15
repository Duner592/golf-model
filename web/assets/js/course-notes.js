(function (global) {
    const MISSING_VALUE = 'N/A';
    const FIELD_DEFINITIONS = [
        { key: 'course', label: 'Course', aliases: ['Course'] },
        { key: 'length', label: 'Length', aliases: ['Lengths', 'Length'] },
        { key: 'par', label: 'Par', aliases: ['Par'] },
        { key: 'course_make_up', label: 'Course Make Up', aliases: ['Course make up', 'Course makeup', 'Course make-up'] },
        { key: 'course_type', label: 'Course Type', aliases: ['Course type', 'Course Type'] },
        { key: 'grass_type', label: 'Grass Type', aliases: ['Grass type', 'Grass Type'] },
        { key: 'course_overview', label: 'Course Overview', aliases: ['Course Overview', 'Course overview'] },
        { key: 'key_attribute', label: 'Key Attribute', aliases: ['Key attribute', 'Key Attribute', 'Key attributes', 'Key Attributes'] },
        { key: 'what_will_it_take_to_win', label: 'What will it take to win?', aliases: ['What will it take to win?', 'What will it take to win'] },
        { key: 'insight', label: 'Insight', aliases: ['Insight', 'Insights'] },
    ];
    const IGNORED_FIELD_ALIASES = ['Time difference', 'Time Difference'];

    function escapeHtml(value) {
        if (value == null) return '';
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeRegex(value) {
        return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function normalizeText(value) {
        return String(value || '')
            .replace(/\u00a0/g, ' ')
            .replace(/\r\n?/g, '\n');
    }

    function cleanValue(value) {
        return normalizeText(value)
            .replace(/^[\s:;\t\-–—]+/, '')
            .replace(/[ \t]+\n/g, '\n')
            .replace(/\n[ \t]+/g, '\n')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    function normalizedForMatch(value) {
        return String(value || '')
            .toLowerCase()
            .replace(/&/g, ' and ')
            .replace(/[^a-z0-9]+/g, ' ')
            .trim();
    }

    function aliasRegex(alias) {
        const labelPattern = escapeRegex(alias).replace(/\\ /g, '[\\s\\u00a0]+');
        const courseGuard = alias.toLowerCase() === 'course'
            ? '(?![\\s\\u00a0]+(?:overview|type|make[\\s\\u00a0]*up))'
            : '';
        return new RegExp(`(^|\\n)\\s*${labelPattern}${courseGuard}(?=$|[\\s\\u00a0:;\\t\\-–—])\\s*(?:[:;\\t\\-–—]\\s*)?`, 'ig');
    }

    function findFieldMatches(rawText) {
        const text = normalizeText(rawText);
        const matches = [];
        const seen = new Set();
        FIELD_DEFINITIONS.forEach(field => {
            field.aliases.forEach(alias => {
                const regex = aliasRegex(alias);
                let match;
                while ((match = regex.exec(text)) !== null) {
                    const prefix = match[1] || '';
                    const start = match.index + prefix.length;
                    const marker = `${field.key}:${start}`;
                    if (seen.has(marker)) {
                        if (match.index === regex.lastIndex) regex.lastIndex += 1;
                        continue;
                    }
                    seen.add(marker);
                    matches.push({
                        key: field.key,
                        start,
                        valueStart: match.index + match[0].length,
                    });
                    if (match.index === regex.lastIndex) regex.lastIndex += 1;
                }
            });
        });
        IGNORED_FIELD_ALIASES.forEach(alias => {
            const regex = aliasRegex(alias);
            let match;
            while ((match = regex.exec(text)) !== null) {
                const prefix = match[1] || '';
                const start = match.index + prefix.length;
                const marker = `ignored:${start}`;
                if (!seen.has(marker)) {
                    seen.add(marker);
                    matches.push({
                        key: null,
                        start,
                        valueStart: match.index + match[0].length,
                    });
                }
                if (match.index === regex.lastIndex) regex.lastIndex += 1;
            }
        });
        return matches.sort((a, b) => a.start - b.start || a.valueStart - b.valueStart);
    }

    function parseDetails(rawDetails, fallbacks = {}) {
        const text = normalizeText(rawDetails);
        const parsed = {};
        const matches = findFieldMatches(text);

        FIELD_DEFINITIONS.forEach(field => {
            parsed[field.key] = '';
        });

        matches.forEach((match, index) => {
            if (!match.key) return;
            if (parsed[match.key]) return;
            const next = matches[index + 1];
            const end = next ? next.start : text.length;
            parsed[match.key] = cleanValue(text.slice(match.valueStart, end));
        });

        FIELD_DEFINITIONS.forEach(field => {
            if (!parsed[field.key]) {
                parsed[field.key] = cleanValue(fallbacks[field.key] || fallbacks[field.label] || '');
            }
            if (!parsed[field.key]) parsed[field.key] = MISSING_VALUE;
        });

        return parsed;
    }

    function parseRow(row) {
        return parseDetails(row && row['Course Details'], {
            course: row && row['Course'],
        });
    }

    function scoreMatch(row, criteria = {}) {
        if (!row || !cleanValue(row['Course Details'])) return 0;
        const course = normalizedForMatch(criteria.course);
        const tournament = normalizedForMatch(criteria.tournament);
        const tour = normalizedForMatch(criteria.tour);
        const rowCourse = normalizedForMatch(row['Course']);
        const rowTournament = normalizedForMatch(row['Tournament']);
        const rowTour = normalizedForMatch(row['Tour/Majors']);
        let score = 0;

        if (course && rowCourse) {
            if (course === rowCourse) score += 100;
            else if (course.includes(rowCourse) || rowCourse.includes(course)) score += 70;
        }
        if (tournament && rowTournament) {
            if (tournament === rowTournament) score += 50;
            else if (tournament.includes(rowTournament) || rowTournament.includes(tournament)) score += 30;
        }
        if (tour && rowTour) {
            if (tour === rowTour) score += 5;
            else if (tour.includes(rowTour) || rowTour.includes(tour)) score += 2;
        }

        return score;
    }

    function findBestRow(rows, criteria = {}) {
        if (!Array.isArray(rows)) return null;
        let bestRow = null;
        let bestScore = 0;
        rows.forEach(row => {
            const score = scoreMatch(row, criteria);
            if (score > bestScore) {
                bestRow = row;
                bestScore = score;
            }
        });
        return bestScore > 0 ? bestRow : null;
    }

    function render(parsedFields) {
        const fields = parsedFields || {};
        return `<dl class="course-notes-list">${
            FIELD_DEFINITIONS.map(field => {
                const value = fields[field.key] || MISSING_VALUE;
                const missingClass = value === MISSING_VALUE ? ' course-note-value--missing' : '';
                return `<div class="course-note-row"><dt><strong>${escapeHtml(field.label)}:</strong></dt><dd class="course-note-value${missingClass}">${escapeHtml(value).replace(/\n/g, '<br>')}</dd></div>`;
            }).join('')
        }</dl>`;
    }

    global.CourseNotes = {
        fields: FIELD_DEFINITIONS,
        missingValue: MISSING_VALUE,
        parseDetails,
        parseRow,
        findBestRow,
        render,
    };
})(window);
