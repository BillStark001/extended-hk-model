/**
 * 
 * const exporter = new ExportUtils();
 * exporter.exportToPDF('main-container', 'my-file.pdf');
 * exporter.exportToSVG('mainSVG', 'my-file.svg');
 */
class ExportUtils {
    constructor() {
        this.isLoading = false;
        this.loadingButtons = new Set();
    }

    /**
     */
    setButtonLoading(buttonSelector, loading, loadingText = '处理中...') {
        const button = document.querySelector(buttonSelector);
        if (!button) return;

        if (loading) {
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.textContent;
            }
            button.textContent = loadingText;
            button.disabled = true;
            this.loadingButtons.add(buttonSelector);
        } else {
            button.textContent = button.dataset.originalText || button.textContent;
            button.disabled = false;
            this.loadingButtons.delete(buttonSelector);
        }
    }

    /**
     */
    async exportToPDF(elementId, filename = 'export.pdf', options = {}) {
        if (this.isLoading) {
            console.warn('PDF导出正在进行中，请稍候...');
            return;
        }

        const element = document.getElementById(elementId);
        if (!element) {
            console.error('找不到要导出的元素:', elementId);
            alert('找不到要导出的内容');
            return;
        }

        this.isLoading = true;
        this.setButtonLoading('.save-btn, .export-button, [onclick*="PDF"], [onclick*="pdf"]', true, '导出PDF中...');

        try {

            const defaultOptions = {
                scale: 2,
                useCORS: true,
                allowTaint: true,
                backgroundColor: '#ffffff',
                logging: false,
                orientation: 'landscape',
                format: 'a4',
                unit: 'mm',
                multiPage: true,
            };

            const finalOptions = { ...defaultOptions, ...options };


            const canvas = await html2canvas(element, {
                scale: finalOptions.scale,
                useCORS: finalOptions.useCORS,
                allowTaint: finalOptions.allowTaint,
                backgroundColor: finalOptions.backgroundColor,
                logging: finalOptions.logging
            });


            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF({
                orientation: finalOptions.orientation,
                unit: finalOptions.unit,
                format: finalOptions.format
            });


            const pageWidth = finalOptions.format === 'a4' && finalOptions.orientation === 'landscape' ? 297 : 210;
            const pageHeight = finalOptions.format === 'a4' && finalOptions.orientation === 'landscape' ? 210 : 297;
            
            if (finalOptions.format === 'a3') {
                const a3Width = finalOptions.orientation === 'landscape' ? 420 : 297;
                const a3Height = finalOptions.orientation === 'landscape' ? 297 : 420;
                var imgWidth = a3Width;
                var maxHeight = a3Height;
            } else {
                var imgWidth = pageWidth;
                var maxHeight = pageHeight;
            }

            const imgHeight = (canvas.height * imgWidth) / canvas.width;


            if (imgHeight > maxHeight && finalOptions.multiPage) {
                let position = 0;
                while (position < imgHeight) {
                    if (position > 0) {
                        pdf.addPage();
                    }
                    
                    const remainingHeight = imgHeight - position;
                    const currentHeight = Math.min(maxHeight, remainingHeight);
                    
                    pdf.addImage(
                        canvas.toDataURL('image/png'),
                        'PNG',
                        0,
                        -position,
                        imgWidth,
                        imgHeight
                    );
                    
                    position += maxHeight;
                }
            } else {
                pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, (maxHeight - imgHeight) / 2, imgWidth, imgHeight);
            }

            pdf.save(filename);
            console.log(`PDF exported successfully: ${filename}`);

        } catch (error) {
            console.error('PDF导出失败:', error);
            alert('PDF导出失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('.save-btn, .export-button, [onclick*="PDF"], [onclick*="pdf"]', false);
            this.isLoading = false;
        }
    }

    /**
     */
    async exportToSVG(svgElementId, filename = 'export.svg', options = {}) {
        if (this.isLoading) {
            console.warn('SVG导出正在进行中，请稍候...');
            return;
        }

        const svgElement = document.getElementById(svgElementId);
        if (!svgElement || svgElement.tagName.toLowerCase() !== 'svg') {
            console.error('找不到SVG元素或元素不是SVG:', svgElementId);
            alert('找不到要导出的SVG内容');
            return;
        }

        this.isLoading = true;
        this.setButtonLoading('[onclick*="SVG"], [onclick*="svg"]', true, '导出SVG中...');

        try {

            const clonedSvg = svgElement.cloneNode(true);
            

            clonedSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            clonedSvg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
            

            if (!clonedSvg.getAttribute('viewBox')) {
                const width = clonedSvg.getAttribute('width') || svgElement.getBoundingClientRect().width;
                const height = clonedSvg.getAttribute('height') || svgElement.getBoundingClientRect().height;
                clonedSvg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            }


            this.inlineStyles(clonedSvg);


            const svgString = new XMLSerializer().serializeToString(clonedSvg);
            const fullSvgString = `<?xml version="1.0" encoding="UTF-8"?>\n${svgString}`;


            const blob = new Blob([fullSvgString], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            

            URL.revokeObjectURL(url);
            
            console.log(`SVG exported successfully: ${filename}`);

        } catch (error) {
            console.error('SVG导出失败:', error);
            alert('SVG导出失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('[onclick*="SVG"], [onclick*="svg"]', false);
            this.isLoading = false;
        }
    }

    /**
     */
    inlineStyles(element) {
        const computedStyle = window.getComputedStyle(element);
        const styleString = Array.from(computedStyle).reduce((str, property) => {
            return `${str}${property}:${computedStyle.getPropertyValue(property)};`;
        }, '');
        
        element.setAttribute('style', styleString);


        Array.from(element.children).forEach(child => {
            this.inlineStyles(child);
        });
    }

    /**
     */
    async exportSVGToPDF(svgElementId, filename = 'export.pdf', options = {}) {
        if (this.isLoading) {
            console.warn('SVG转PDF正在进行中，请稍候...');
            return;
        }

        const svgElement = document.getElementById(svgElementId);
        if (!svgElement || svgElement.tagName.toLowerCase() !== 'svg') {
            console.error('找不到SVG元素或元素不是SVG:', svgElementId);
            alert('找不到要导出的SVG内容');
            return;
        }


        if (typeof window.svg2pdf === 'undefined') {
            console.warn('svg2pdf.js not available, falling back to html2canvas method');
            return this.exportToPDF(svgElementId, filename, options);
        }

        this.isLoading = true;
        this.setButtonLoading('[onclick*="PDF"], [onclick*="pdf"]', true, '导出PDF中...');

        try {
            const { jsPDF } = window.jspdf;
            const defaultOptions = {
                orientation: 'landscape',
                unit: 'pt',
                format: [600, 450]
            };

            const finalOptions = { ...defaultOptions, ...options };
            const pdf = new jsPDF(finalOptions);

            await window.svg2pdf(svgElement, pdf, {
                xOffset: 0,
                yOffset: 0,
                scale: 1
            });

            pdf.save(filename);
            console.log(`SVG to PDF exported successfully: ${filename}`);

        } catch (error) {
            console.error('SVG转PDF失败:', error);
            alert('SVG转PDF失败，请查看控制台了解详细错误信息。');
        } finally {
            this.setButtonLoading('[onclick*="PDF"], [onclick*="pdf"]', false);
            this.isLoading = false;
        }
    }

    /**
     */
    async exportMultipleFormats(elementId, baseName = 'export', formats = ['pdf', 'svg'], options = {}) {
        if (this.isLoading) {
            console.warn('批量导出正在进行中，请稍候...');
            return;
        }

        for (const format of formats) {
            try {
                switch (format.toLowerCase()) {
                    case 'pdf':
                        await this.exportToPDF(elementId, `${baseName}.pdf`, options.pdf || {});
                        break;
                    case 'svg':
                        await this.exportToSVG(elementId, `${baseName}.svg`, options.svg || {});
                        break;
                    case 'svgpdf':
                        await this.exportSVGToPDF(elementId, `${baseName}_vector.pdf`, options.svgpdf || {});
                        break;
                    default:
                        console.warn(`Unsupported format: ${format}`);
                }
                

                await new Promise(resolve => setTimeout(resolve, 500));
            } catch (error) {
                console.error(`Error exporting ${format}:`, error);
            }
        }
    }

    /**
     */
    checkDependencies() {
        const dependencies = {
            jsPDF: typeof window.jspdf !== 'undefined',
            html2canvas: typeof window.html2canvas !== 'undefined',
            svg2pdf: typeof window.svg2pdf !== 'undefined'
        };

        console.log('Export dependencies status:', dependencies);
        return dependencies;
    }
}


window.ExportUtils = ExportUtils;


window.exportUtils = new ExportUtils();