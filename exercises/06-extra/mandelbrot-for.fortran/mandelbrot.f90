program mandelbrot
        use omp_lib
	implicit none
	integer, parameter :: width = 1920
	integer, parameter :: height = 1080
	integer, parameter :: max_iter = 8000
	double precision, parameter :: x_min = -2.5
	double precision, parameter :: x_max = 1.5
	double precision, parameter :: pi = 3.1415927
	character(len=7) :: output = "out.png"
	character, dimension(0:width * height * 4) :: image
	double precision :: rx, ry, zx, zy, zx2, zy2, as, y_min, y_max, pw, ph
	integer :: x, y, i, j
        double precision :: starttime

	as = -(height/real(width)) * (x_max - x_min) /2
	y_min = -as
	y_max = as
	pw = (x_max - x_min) / width
	ph = (y_max - y_min) / height
        starttime = omp_get_wtime()

	!$omp parallel do shared(image) default(shared) 
	do j = 0, height * width -1
		x = mod(j, width)
		y = j / width
		ry = y_min + y * ph
		rx = x_min + x * pw
		zx = 0.0
		zy = 0.0
		zx2 = 0.0
		zy2 = 0.0
		do i = 0, MAX_ITER
			if ((zx2 + zy2) >= 4) exit
			zy = 2 * zx * zy + ry
			zx = zx2 - zy2 + rx
			zx2 = zx * zx
                        zy2 = zy * zy
		end do
		image(j * 4) = 		achar(255 - int((cos(i * pi / real(max_iter)) + 1)/2 * 255))
		image(j * 4 + 1) = 	achar(255 - int((sin(i * pi / real(max_iter)) + 1)/3 * 255))
		image(j * 4 + 2) = 	achar(255 - int(i / real(max_iter) * 255))
		image(j * 4 + 3) = 	achar(255)
	end do
	!$omp end parallel do
        write (*,*) "Calculation took", (omp_get_wtime()-starttime), "seconds using", omp_get_max_threads(), "threads."
	call lodepng_encode32_file(output, image, width, height)
end program
